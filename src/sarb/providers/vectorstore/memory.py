"""An exhaustive vector store, in this process. No Docker, no server, no network.

This is a **real store**, not a stub. It does exact nearest-neighbour search over every
record it holds and a complete evaluation of the filter algebra — so an engine tested
against it is tested against real geometry, and a retrieval test that asserts an ordering
is asserting something true.

Two consequences follow, and both are the reason this file is worth its length.

It is the **reference implementation of the filter algebra**. Qdrant compiles a filter to
Qdrant's DSL and Postgres compiles it to SQL, and either compiler can be subtly wrong in
a way that no unit test of the compiler itself would catch — because the test would be
checking the compiler against the same misunderstanding that produced it. Here the
semantics are just Python, readable in one sitting, and the conformance suite runs the
same filters against all three backends and demands the same answers.

And it makes a laptop enough. ``provider: memory`` in the ``vector_stores`` block, and a
whole RAG system runs with nothing installed. Exhaustive search is O(n) per query, which
is fine for the thousands of vectors a developer has and hopeless for the millions a
production system has — so it is deliberately not something you can point at production
by accident.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
from sarb.errors import VectorStoreError
from sarb.providers.vectorstore._shared import (
    VectorStoreOptions,
    check_delete_selector,
    check_dimensions,
    cosine,
    score_from_distance,
)

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from sarb.plugins.registry import ComponentContext

__all__ = ["MemoryVectorStore", "MemoryVectorStoreOptions", "build", "matches"]

PROVIDER = "memory"


class MemoryVectorStoreOptions(VectorStoreOptions):
    """Options for the in-process store. There is nothing to configure, and that is the point."""


@dataclass
class _Collection:
    dimensions: int
    distance: Distance
    records: dict[str, VectorRecord] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The filter algebra, evaluated
# --------------------------------------------------------------------------- #


def matches(metadata: dict[str, Any], where: Filter | None) -> bool:
    """Whether a record's metadata satisfies a filter.

    The reference semantics. Every other adapter's compiler is checked against this.
    """
    if where is None:
        return True

    match where.kind:
        case "and":
            return all(matches(metadata, f) for f in where.filters)
        case "or":
            return any(matches(metadata, f) for f in where.filters)
        case "not":
            return not matches(metadata, where.filter)
        case _:
            return _compare(metadata, where)


#: Every comparison but `exists`, which is about presence rather than value and so is
#: handled before this table is reached.
_OPS: dict[CompareOp, Callable[[Any, Any], bool]] = {
    CompareOp.EQ: operator.eq,
    CompareOp.NE: operator.ne,
    CompareOp.GT: operator.gt,
    CompareOp.GTE: operator.ge,
    CompareOp.LT: operator.lt,
    CompareOp.LTE: operator.le,
    CompareOp.IN: lambda actual, value: actual in _as_list(value),
    CompareOp.NOT_IN: lambda actual, value: actual not in _as_list(value),
}


def _compare(metadata: dict[str, Any], where: Compare) -> bool:
    """One comparison against one field."""
    present = where.field in metadata

    if where.op is CompareOp.EXISTS:
        return present is bool(where.value)

    # An absent field satisfies nothing — not even `ne`. This is the decision most
    # likely to surprise, so it is made explicitly rather than falling out of Python's
    # comparison rules: `ne("tenant", "acme")` must not match a record that has no
    # tenant at all, or a filter meant to *exclude* one tenant silently admits every
    # record that forgot to say which tenant it belonged to.
    if not present:
        return False

    try:
        return bool(_OPS[where.op](metadata[where.field], where.value))
    except TypeError:
        # `gte("year", 2020)` against a record whose `year` is the string "2020".
        # Python raises; Postgres would raise; Qdrant would silently not match. We
        # follow the majority, and treat it as "does not match" rather than an error —
        # one badly-typed record in a corpus of a million must not fail the query.
        return False


def _as_list(value: Any) -> list[Any]:
    """`in` and `not_in` are validated to hold lists by the contract."""
    return value if isinstance(value, list) else [value]


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


class MemoryVectorStore:
    """The :class:`~sarb.contracts.VectorStore` port, held in a dict."""

    def __init__(self, options: MemoryVectorStoreOptions) -> None:
        self._options = options
        self._collections: dict[str, _Collection] = {}

    async def ensure_collection(
        self,
        collection: str,
        *,
        dimensions: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Create the collection if absent; refuse to reshape it if present."""
        existing = self._collections.get(collection)

        if existing is None:
            self._collections[collection] = _Collection(dimensions, distance)
            return

        if existing.dimensions != dimensions or existing.distance != distance:
            raise VectorStoreError(
                f"collection {collection!r} already exists as "
                f"{existing.dimensions}-dimensional {existing.distance.value}, and cannot be "
                f"reshaped to {dimensions}-dimensional {distance.value}. Drop it and rebuild.",
                provider=PROVIDER,
                context={"collection": collection},
            )

    async def drop_collection(self, collection: str) -> None:
        """Delete the collection and everything in it. Idempotent."""
        self._collections.pop(collection, None)

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        """Insert or replace by id."""
        target = self._require(collection)
        check_dimensions(
            records, dimensions=target.dimensions, collection=collection, provider=PROVIDER
        )

        for record in records:
            target.records[record.id] = record

    async def search(self, collection: str, query: VectorQuery) -> Sequence[SearchHit]:
        """Exhaustive search. Every record, scored, sorted, cut."""
        target = self._require(collection)

        if len(query.vector) != target.dimensions:
            raise VectorStoreError(
                f"query vector is {len(query.vector)}-dimensional, but collection "
                f"{collection!r} holds {target.dimensions}-dimensional ones",
                provider=PROVIDER,
                context={"collection": collection},
            )

        hits = [
            SearchHit(
                id=record.id,
                score=self._score(query.vector, record.vector, target.distance),
                metadata=record.metadata,
                vector=record.vector if query.include_vectors else None,
            )
            # The filter is applied *before* the cut, not after. Filtering the top-k
            # would silently return fewer than k results — the single most common bug
            # in hand-rolled vector search, and one that looks like "retrieval is a bit
            # weak" rather than like a bug.
            for record in target.records.values()
            if matches(record.metadata, query.where)
        ]

        hits.sort(key=lambda hit: hit.score, reverse=True)

        if query.min_score is not None:
            hits = [hit for hit in hits if hit.score >= query.min_score]

        return hits[: query.top_k]

    async def fetch(self, collection: str, ids: Sequence[str]) -> Sequence[VectorRecord]:
        """Get by id. Missing ids are omitted, not raised."""
        target = self._require(collection)
        return [target.records[i] for i in ids if i in target.records]

    async def delete(
        self,
        collection: str,
        *,
        ids: Sequence[str] | None = None,
        where: Filter | None = None,
    ) -> int:
        """Delete by id or by filter. Exactly one."""
        check_delete_selector(ids, where, provider=PROVIDER)
        target = self._require(collection)

        doomed = (
            [i for i in ids if i in target.records]
            if ids is not None
            else [i for i, r in target.records.items() if matches(r.metadata, where)]
        )

        for record_id in doomed:
            del target.records[record_id]

        return len(doomed)

    async def count(self, collection: str, *, where: Filter | None = None) -> int:
        """How many records match."""
        target = self._require(collection)

        if where is None:
            return len(target.records)

        return sum(1 for r in target.records.values() if matches(r.metadata, where))

    # -- internals ---------------------------------------------------------- #

    def _require(self, collection: str) -> _Collection:
        """The collection, or a message that says how to create it."""
        target = self._collections.get(collection)
        if target is None:
            raise VectorStoreError(
                f"no collection {collection!r}. Call ensure_collection first.",
                provider=PROVIDER,
                context={"collection": collection, "known": sorted(self._collections)},
            )
        return target

    @staticmethod
    def _score(query: Sequence[float], stored: Sequence[float], distance: Distance) -> float:
        """Similarity, with higher always better — the port's promise, kept here too."""
        match distance:
            case Distance.COSINE:
                return cosine(query, stored)
            case Distance.DOT:
                return sum(x * y for x, y in zip(query, stored, strict=True))
            case _:
                euclidean = sum((x - y) ** 2 for x, y in zip(query, stored, strict=True)) ** 0.5
                return score_from_distance(euclidean, Distance.EUCLIDEAN)


def build(ctx: ComponentContext) -> MemoryVectorStore:
    """Build the ``memory`` vector store from configuration."""
    return MemoryVectorStore(
        parse_options(ctx.config, MemoryVectorStoreOptions, component=ctx.path)
    )
