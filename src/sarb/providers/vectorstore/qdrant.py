"""Qdrant — a database built for this and nothing else.

Two things in here are worth reading before writing another vector-store adapter,
because they are the two places the abstraction actually costs something.

The filter compiler
    Our filter algebra becomes Qdrant's ``Filter``/``FieldCondition`` DSL. Anything it
    cannot express raises, and the reason that rule exists is worth restating: a dropped
    filter returns **more** rows, not fewer. It fails *open*. A ``tenant`` filter that
    is quietly ignored does not error, does not log, and does not return nothing — it
    returns another tenant's documents, ranked by relevance, and hands them to whoever
    asked.

The ids
    Qdrant point ids must be an unsigned 64-bit integer or a UUID. An id like
    ``"contract-42#chunk-3"`` — which is exactly what a chunker produces — passes the
    client's type check and is then rejected by the server. So the adapter mints a
    deterministic UUIDv5 from the caller's id and keeps the original in the payload. See
    :func:`point_id`.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field, SecretStr

from sarb.config import parse_options
from sarb.contracts import (
    Compare,
    CompareOp,
    Distance,
    Filter,
    SearchHit,
    VectorQuery,
    VectorRecord,
)
from sarb.dependencies import import_optional
from sarb.errors import SarbError, VectorStoreError
from sarb.providers._shared import unsupported
from sarb.providers.vectorstore._shared import (
    VectorStoreOptions,
    check_delete_selector,
    check_dimensions,
    score_from_distance,
)

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from qdrant_client import AsyncQdrantClient

    from sarb.plugins.registry import ComponentContext

__all__ = ["QdrantOptions", "QdrantStore", "build", "point_id"]

PROVIDER = "qdrant"
EXTRA = "qdrant"

#: The payload key holding the caller's real id. Namespaced so it cannot collide with a
#: metadata field somebody legitimately called "id".
ID_KEY = "__sarb_id"

#: Namespace for minting point ids. Fixed forever: change it and every id in every
#: existing collection changes with it, which means every re-index writes a duplicate
#: instead of a replacement, silently doubling the corpus.
ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def point_id(record_id: str) -> str:
    """The UUID Qdrant will accept for a caller's id.

    Deterministic, because upserting the same id must *replace* the record rather than
    add a second copy of it — which is the property that makes re-indexing a changed
    document idempotent, and the property a random id would destroy.
    """
    return str(uuid.uuid5(ID_NAMESPACE, record_id))


class QdrantOptions(VectorStoreOptions):
    """Options for the Qdrant adapter.

    Attributes:
        url: ``http://localhost:6333``, or a Qdrant Cloud endpoint.
        api_key: Required by Qdrant Cloud, absent for a local one.
        prefer_grpc: Faster for large batches. HTTP is easier to debug.
        timeout: Per-request deadline, in seconds.
    """

    url: str = "http://localhost:6333"
    api_key: SecretStr | None = None
    prefer_grpc: bool = False
    timeout: int = Field(default=30, gt=0)


class QdrantStore:
    """The :class:`~sarb.contracts.VectorStore` port, backed by Qdrant.

    The client is injected, so the translation in here is testable against Qdrant's own
    models with no server running. :func:`build` constructs the client.
    """

    _DISTANCE: ClassVar[dict[Distance, str]] = {
        Distance.COSINE: "Cosine",
        Distance.DOT: "Dot",
        Distance.EUCLIDEAN: "Euclid",
    }

    def __init__(self, client: AsyncQdrantClient, options: QdrantOptions) -> None:
        self._client = client
        self._options = options
        self._models = import_optional(
            "qdrant_client.models", extra=EXTRA, used_by=f"vector store {PROVIDER!r}"
        )
        #: Collection -> (dimensions, distance), learned at ensure_collection. Lets a
        #: mis-sized write fail with a useful message before a network round trip.
        self._shapes: dict[str, tuple[int, Distance]] = {}

    # -- collections -------------------------------------------------------- #

    async def ensure_collection(
        self,
        collection: str,
        *,
        dimensions: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Create the collection if absent; refuse to reshape it if present."""
        m = self._models

        try:
            if await self._client.collection_exists(collection):
                await self._check_shape(collection, dimensions, distance)
            else:
                await self._client.create_collection(
                    collection_name=collection,
                    vectors_config=m.VectorParams(
                        size=dimensions, distance=self._DISTANCE[distance]
                    ),
                )
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        self._shapes[collection] = (dimensions, distance)

    async def _check_shape(self, collection: str, dimensions: int, distance: Distance) -> None:
        """An existing collection must be the shape we were asked for."""
        info = await self._client.get_collection(collection)
        params = info.config.params.vectors

        actual_size = getattr(params, "size", None)
        actual_distance = getattr(params, "distance", None)

        if actual_size != dimensions or str(actual_distance) != self._DISTANCE[distance]:
            raise VectorStoreError(
                f"collection {collection!r} already exists as {actual_size}-dimensional "
                f"{actual_distance}, and cannot be reshaped to {dimensions}-dimensional "
                f"{self._DISTANCE[distance]}. Drop it and rebuild the index.",
                provider=PROVIDER,
                context={"collection": collection},
            )

    async def drop_collection(self, collection: str) -> None:
        """Delete the collection and everything in it. Idempotent."""
        try:
            await self._client.delete_collection(collection)
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        self._shapes.pop(collection, None)

    # -- writing ------------------------------------------------------------ #

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        """Insert or replace by id, in batches the server will accept."""
        if not records:
            return

        shape = self._shapes.get(collection)
        if shape:
            check_dimensions(records, dimensions=shape[0], collection=collection, provider=PROVIDER)

        m = self._models
        size = self._options.batch_size

        try:
            for start in range(0, len(records), size):
                batch = records[start : start + size]
                await self._client.upsert(
                    collection_name=collection,
                    points=[
                        m.PointStruct(
                            id=point_id(record.id),
                            vector=list(record.vector),
                            payload={**record.metadata, ID_KEY: record.id},
                        )
                        for record in batch
                    ],
                )
        except Exception as exc:
            raise self._fail(exc, collection) from exc

    async def delete(
        self,
        collection: str,
        *,
        ids: Sequence[str] | None = None,
        where: Filter | None = None,
    ) -> int:
        """Delete by id or by filter. Exactly one."""
        check_delete_selector(ids, where, provider=PROVIDER)
        m = self._models

        # Qdrant's delete does not report how many points it removed, so the count is
        # taken first. Racy under concurrent writes, and honest about it — the
        # alternative is inventing a number, and a wrong count is worse than a stale one.
        try:
            if ids is not None:
                found = await self._client.retrieve(collection, ids=[point_id(i) for i in ids])
                removed = len(found)
                await self._client.delete(
                    collection_name=collection,
                    points_selector=m.PointIdsList(points=[point_id(i) for i in ids]),
                )
            else:
                removed = await self.count(collection, where=where)
                await self._client.delete(
                    collection_name=collection,
                    points_selector=m.FilterSelector(filter=self.compile(where)),
                )
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        return removed

    # -- reading ------------------------------------------------------------ #

    async def search(self, collection: str, query: VectorQuery) -> Sequence[SearchHit]:
        """Search, with the filter applied *by the index* rather than after it."""
        distance = self._shapes.get(collection, (0, Distance.COSINE))[1]

        try:
            response = await self._client.query_points(
                collection_name=collection,
                query=list(query.vector),
                limit=query.top_k,
                # Not a post-filter. Filtering the top-k after the fact silently returns
                # fewer than top_k, which reads as "retrieval is a bit weak" rather than
                # as a bug, and is therefore never found.
                query_filter=self.compile(query.where),
                score_threshold=query.min_score,
                with_payload=True,
                with_vectors=query.include_vectors,
            )
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        return [self._hit(point, distance) for point in response.points]

    async def fetch(self, collection: str, ids: Sequence[str]) -> Sequence[VectorRecord]:
        """Get by id. Missing ids are omitted, not raised."""
        if not ids:
            return []

        try:
            points = await self._client.retrieve(
                collection_name=collection,
                ids=[point_id(i) for i in ids],
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        return [
            VectorRecord(
                id=self._original_id(point.payload, point.id),
                vector=list(point.vector or []),
                metadata=self._metadata(point.payload),
            )
            for point in points
        ]

    async def count(self, collection: str, *, where: Filter | None = None) -> int:
        """How many records match."""
        try:
            result = await self._client.count(
                collection_name=collection,
                count_filter=self.compile(where),
                exact=True,
            )
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        return int(result.count)

    # -- the filter compiler ------------------------------------------------ #

    def compile(self, where: Filter | None) -> Any:
        """Compile our filter algebra into Qdrant's.

        Returns ``None`` for no filter, which Qdrant reads as "everything".
        """
        if where is None:
            return None

        return self._models.Filter(must=[self._condition(where)])

    def _condition(self, where: Filter) -> Any:
        """One node of the tree."""
        m = self._models

        match where.kind:
            case "and":
                return m.Filter(must=[self._condition(f) for f in where.filters])
            case "or":
                return m.Filter(should=[self._condition(f) for f in where.filters])
            case "not":
                return m.Filter(must_not=[self._condition(where.filter)])
            case _:
                return self._compare(where)

    def _compare(self, where: Compare) -> Any:
        """One comparison. Everything the algebra can say, Qdrant can hear."""
        m = self._models
        field = where.field

        match where.op:
            case CompareOp.EQ:
                return m.FieldCondition(key=field, match=m.MatchValue(value=where.value))
            case CompareOp.NE:
                # `must_not` alone would also match records that have no such field at
                # all — a record with no `tenant` is, after all, not equal to "acme".
                # The algebra says otherwise: a *comparison* against a field that is not
                # there is false, so `ne` means "has the field, and it differs". The
                # empty condition is what makes Qdrant agree with the reference
                # implementation in `memory.py`, and the conformance suite is what
                # noticed that it did not.
                return m.Filter(
                    must_not=[
                        m.FieldCondition(key=field, match=m.MatchValue(value=where.value)),
                        m.IsEmptyCondition(is_empty=m.PayloadField(key=field)),
                    ]
                )
            case CompareOp.IN:
                return m.FieldCondition(key=field, match=m.MatchAny(any=where.value))
            case CompareOp.NOT_IN:
                # Same presence guard, same reason.
                return m.Filter(
                    must_not=[
                        m.FieldCondition(key=field, match=m.MatchAny(any=where.value)),
                        m.IsEmptyCondition(is_empty=m.PayloadField(key=field)),
                    ]
                )
            case CompareOp.EXISTS:
                # Qdrant has no "has key". It has "is empty", so presence is its
                # negation — and absence is the condition itself.
                empty = m.IsEmptyCondition(is_empty=m.PayloadField(key=field))
                return m.Filter(must_not=[empty]) if where.value else m.Filter(must=[empty])
            case _:
                # The four ordering comparisons. Our operator names are Qdrant's `Range`
                # field names — `gt`, `gte`, `lt`, `lte` — so the op *is* the keyword.
                bound = {str(where.op): self._number(where)}
                return m.FieldCondition(key=field, range=m.Range(**bound))

    @staticmethod
    def _number(where: Compare) -> float:
        """Ordering comparisons need a number. Qdrant's ``Range`` cannot hold anything else.

        This is the one place the algebra is wider than the backend. ``lt("name", "m")``
        is expressible in SQL and meaningless to Qdrant, so it raises here rather than
        being sent and silently matching nothing — which would look exactly like "there
        are no such documents".
        """
        if isinstance(where.value, bool) or not isinstance(where.value, int | float):
            raise unsupported(
                PROVIDER,
                f"{where.op.value} on the non-numeric value {where.value!r}",
                hint="Qdrant can only order numbers. Use eq/in for strings.",
            )
        return float(where.value)

    # -- translation back --------------------------------------------------- #

    def _hit(self, point: Any, distance: Distance) -> SearchHit:
        """One scored point, with the score pointing the way the port promises."""
        return SearchHit(
            id=self._original_id(point.payload, point.id),
            score=score_from_distance(point.score, distance),
            metadata=self._metadata(point.payload),
            vector=list(point.vector) if point.vector else None,
        )

    @staticmethod
    def _original_id(payload: dict[str, Any] | None, fallback: Any) -> str:
        """The caller's id, which we put in the payload because Qdrant would not take it."""
        return str((payload or {}).get(ID_KEY, fallback))

    @staticmethod
    def _metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
        """The payload, minus our own bookkeeping."""
        return {k: v for k, v in (payload or {}).items() if k != ID_KEY}

    # -- failure ------------------------------------------------------------ #

    def _fail(self, exc: Exception, collection: str) -> SarbError:
        """Translate a Qdrant failure into one of ours."""
        if isinstance(exc, SarbError):
            return exc

        return VectorStoreError(
            f"Qdrant call failed: {exc}",
            provider=PROVIDER,
            context={"collection": collection, "url": self._options.url},
            cause=exc,
            # A vector database is a network service. Connection resets, leader
            # elections and restarts are all transient, and the middleware knows what
            # to do with a retryable error.
            retryable=not isinstance(exc, ValueError),
        )

    async def aclose(self) -> None:
        """Close the client's connection pool."""
        await self._client.close()


def build(ctx: ComponentContext) -> QdrantStore:
    """Build the ``qdrant`` vector store from configuration."""
    sdk = import_optional("qdrant_client", extra=EXTRA, used_by=f"vector store {PROVIDER!r}")
    options = parse_options(ctx.config, QdrantOptions, component=ctx.path)

    client = sdk.AsyncQdrantClient(
        url=options.url,
        api_key=options.api_key.get_secret_value() if options.api_key else None,
        prefer_grpc=options.prefer_grpc,
        timeout=options.timeout,
    )
    return QdrantStore(client, options)
