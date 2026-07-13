"""pgvector — vectors in the Postgres you already run.

The pragmatic choice, and often the right one. One database to back up, one to monitor,
one to hold a transaction across: a chunk and its vector can be written in the same
commit as the row that owns them, which no dedicated vector database can offer.

Three things this adapter has to invent, because Postgres does not have them:

A collection
    pgvector has tables, not collections. So a collection is a table, and a small
    registry table remembers each one's width and distance metric — because in pgvector
    the metric lives in the *index operator class* and the *query operator*, not in the
    column, and there is nowhere else to keep it.

Safe identifiers
    A table per collection means the table name is dynamic, and a dynamic identifier is
    a SQL injection surface. Postgres cannot parameterise an identifier, so the name is
    validated against a strict pattern and quoted. See :func:`table_for`.

Three-valued logic
    ``NULL`` is not ``False``. ``metadata->>'tenant' <> 'acme'`` on a row with no
    ``tenant`` is ``NULL``, which is not a match — but ``NOT (metadata->>'tenant' =
    'acme')`` is also ``NULL``, which is *also* not a match, and that disagrees with what
    the filter algebra means. Every comparison therefore carries an explicit presence
    guard so it yields ``FALSE`` rather than ``NULL``, and ``NOT`` negates something real.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
from sarb.errors import InvalidInputError, SarbError, VectorStoreError
from sarb.providers.vectorstore._shared import (
    VectorStoreOptions,
    check_delete_selector,
    check_dimensions,
)

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from sarb.plugins.registry import ComponentContext

__all__ = ["PgVectorOptions", "PgVectorStore", "build", "table_for"]

PROVIDER = "pgvector"
EXTRA = "pgvector"

#: Where each collection's shape is remembered. Postgres has no other place to put it.
REGISTRY = "sarb_collections"

#: A collection name we are willing to interpolate into SQL. Deliberately narrow: this
#: is the one string in the adapter that cannot be a bind parameter, so it is the one
#: string that has to be proven safe rather than escaped and hoped for.
SAFE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,48}$")

#: distance -> (operator, how to turn the operator's number into a score).
#:
#: Every one of these operators returns a *distance* except `<#>`, which returns the
#: negative inner product. Handing any of them back as a "score" would invert the
#: ranking while raising nothing at all — the search would work, and return the least
#: relevant documents first.
_OPERATORS: dict[Distance, str] = {
    Distance.COSINE: "<=>",
    Distance.DOT: "<#>",
    Distance.EUCLIDEAN: "<->",
}

_OPCLASS: dict[Distance, str] = {
    Distance.COSINE: "vector_cosine_ops",
    Distance.DOT: "vector_ip_ops",
    Distance.EUCLIDEAN: "vector_l2_ops",
}


def table_for(collection: str) -> str:
    """The quoted table name for a collection, or a refusal.

    Postgres cannot bind an identifier the way it binds a value, so this string is
    interpolated into SQL — which makes it the only place in the adapter where a
    caller's input reaches the query text. It is therefore checked against
    :data:`SAFE_NAME` rather than escaped: a name that is not obviously safe is not
    used at all.

    Raises:
        InvalidInputError: The collection name is not one we will put in a query.
    """
    if not SAFE_NAME.match(collection):
        raise InvalidInputError(
            f"collection name {collection!r} is not usable as a Postgres table. "
            "Use lowercase letters, digits and underscores, starting with a letter.",
            context={"provider": PROVIDER, "collection": collection},
        )
    return f'"sarb_{collection}"'


class PgVectorOptions(VectorStoreOptions):
    """Options for the pgvector adapter.

    Attributes:
        dsn: ``postgresql://user:pass@host/db``.
        min_size: Minimum pooled connections.
        max_size: Maximum pooled connections.
        ef_search: HNSW search breadth. Higher is more accurate and slower.

    Why ``ef_search`` is exposed:
        An HNSW index walks a graph, and a filtered search walks it looking for rows
        that also satisfy the filter. If the filter is selective — one tenant out of a
        thousand — the walk can exhaust its budget before finding ``top_k`` matches, and
        the query returns **fewer results than it should**, silently. Raising
        ``ef_search`` is the fix, and it has to be reachable from config because the
        right value depends on your data, not on this framework.
    """

    dsn: SecretStr
    min_size: int = Field(default=1, ge=0)
    max_size: int = Field(default=10, gt=0)
    ef_search: int = Field(default=64, gt=0)


@dataclass(frozen=True, slots=True)
class _Sql:
    """A compiled SQL fragment and the values it binds."""

    text: str
    params: list[Any] = field(default_factory=list)


class PgVectorStore:
    """The :class:`~sarb.contracts.VectorStore` port, backed by Postgres.

    The connection pool is injected, so the SQL this adapter generates is testable —
    which is the point, because the SQL *is* the adapter.
    """

    def __init__(self, pool: Any, options: PgVectorOptions) -> None:
        self._pool = pool
        self._options = options
        self._shapes: dict[str, tuple[int, Distance]] = {}

    # -- collections -------------------------------------------------------- #

    async def ensure_collection(
        self,
        collection: str,
        *,
        dimensions: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Create the table, its indexes and its registry row. Idempotent."""
        table = table_for(collection)

        try:
            async with self._pool.acquire() as conn, conn.transaction():
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {REGISTRY} ("
                    "  name text PRIMARY KEY,"
                    "  dimensions integer NOT NULL,"
                    "  distance text NOT NULL)"
                )

                existing = await conn.fetchrow(
                    f"SELECT dimensions, distance FROM {REGISTRY} WHERE name = $1", collection
                )
                if existing:
                    self._check_shape(collection, existing, dimensions, distance)
                    self._shapes[collection] = (dimensions, distance)
                    return

                await conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} ("
                    "  id text PRIMARY KEY,"
                    f"  vector vector({dimensions}) NOT NULL,"
                    "  metadata jsonb NOT NULL DEFAULT '{}'::jsonb)"
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS sarb_{collection}_vector "
                    f"ON {table} USING hnsw (vector {_OPCLASS[distance]})"
                )
                # Without this, every filtered search is a sequential scan of the
                # metadata column, and the filter that was supposed to make the query
                # cheaper makes it the most expensive thing in the system.
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS sarb_{collection}_metadata "
                    f"ON {table} USING gin (metadata)"
                )
                await conn.execute(
                    f"INSERT INTO {REGISTRY} (name, dimensions, distance) VALUES ($1, $2, $3)",
                    collection,
                    dimensions,
                    distance.value,
                )
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        self._shapes[collection] = (dimensions, distance)

    def _check_shape(
        self, collection: str, existing: Any, dimensions: int, distance: Distance
    ) -> None:
        """An existing collection must be the shape we were asked for."""
        if existing["dimensions"] != dimensions or existing["distance"] != distance.value:
            raise VectorStoreError(
                f"collection {collection!r} already exists as {existing['dimensions']}-"
                f"dimensional {existing['distance']}, and cannot be reshaped to "
                f"{dimensions}-dimensional {distance.value}. Drop it and rebuild the index.",
                provider=PROVIDER,
                context={"collection": collection},
            )

    async def drop_collection(self, collection: str) -> None:
        """Delete the table and forget the collection. Idempotent."""
        table = table_for(collection)

        try:
            async with self._pool.acquire() as conn, conn.transaction():
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
                await conn.execute(f"DELETE FROM {REGISTRY} WHERE name = $1", collection)
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        self._shapes.pop(collection, None)

    # -- writing ------------------------------------------------------------ #

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        """Insert or replace by id, in batches, in one transaction each."""
        if not records:
            return

        table = table_for(collection)
        shape = self._shapes.get(collection)
        if shape:
            check_dimensions(records, dimensions=shape[0], collection=collection, provider=PROVIDER)

        statement = (
            f"INSERT INTO {table} (id, vector, metadata) VALUES ($1, $2::vector, $3::jsonb) "
            "ON CONFLICT (id) DO UPDATE SET vector = EXCLUDED.vector, "
            "metadata = EXCLUDED.metadata"
        )
        size = self._options.batch_size

        try:
            async with self._pool.acquire() as conn:
                for start in range(0, len(records), size):
                    batch = records[start : start + size]
                    await conn.executemany(
                        statement,
                        [(r.id, _vector_literal(r.vector), json.dumps(r.metadata)) for r in batch],
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
        table = table_for(collection)

        if ids is not None:
            sql = f"DELETE FROM {table} WHERE id = ANY($1::text[])"
            params: list[Any] = [list(ids)]
        else:
            clause = self.compile(where, start=1)
            sql = f"DELETE FROM {table} WHERE {clause.text}"
            params = clause.params

        try:
            async with self._pool.acquire() as conn:
                status = await conn.execute(sql, *params)
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        # asyncpg returns the command tag, e.g. "DELETE 3". Postgres does report how
        # many rows it removed, so unlike Qdrant this number is exact rather than a
        # count taken beforehand and hoped to still be true.
        return int(str(status).rsplit(" ", maxsplit=1)[-1] or 0)

    # -- reading ------------------------------------------------------------ #

    async def search(self, collection: str, query: VectorQuery) -> Sequence[SearchHit]:
        """Search, with the filter in the WHERE clause and the ordering in the index."""
        table = table_for(collection)
        _, distance = self._shapes.get(collection, (0, Distance.COSINE))
        operator = _OPERATORS[distance]

        params: list[Any] = [_vector_literal(query.vector)]
        where = "TRUE"
        if query.where is not None:
            clause = self.compile(query.where, start=2)
            where = clause.text
            params.extend(clause.params)

        params.append(query.top_k)
        limit = f"${len(params)}"

        sql = (
            f"SELECT id, metadata, {'vector::text,' if query.include_vectors else ''} "
            f"vector {operator} $1::vector AS distance "
            f"FROM {table} WHERE {where} "
            f"ORDER BY vector {operator} $1::vector "
            f"LIMIT {limit}"
        )

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(f"SET LOCAL hnsw.ef_search = {self._options.ef_search}")
                rows = await conn.fetch(sql, *params)
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        hits = [self._hit(row, distance, query.include_vectors) for row in rows]

        if query.min_score is not None:
            hits = [hit for hit in hits if hit.score >= query.min_score]

        return hits

    async def fetch(self, collection: str, ids: Sequence[str]) -> Sequence[VectorRecord]:
        """Get by id. Missing ids are omitted, not raised."""
        if not ids:
            return []

        table = table_for(collection)

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT id, vector::text, metadata FROM {table} WHERE id = ANY($1::text[])",
                    list(ids),
                )
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        return [
            VectorRecord(
                id=row["id"],
                vector=_parse_vector(row["vector"]),
                metadata=_parse_metadata(row["metadata"]),
            )
            for row in rows
        ]

    async def count(self, collection: str, *, where: Filter | None = None) -> int:
        """How many records match."""
        table = table_for(collection)

        sql = f"SELECT count(*) AS n FROM {table}"
        params: list[Any] = []
        if where is not None:
            clause = self.compile(where, start=1)
            sql = f"{sql} WHERE {clause.text}"
            params = clause.params

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(sql, *params)
        except Exception as exc:
            raise self._fail(exc, collection) from exc

        return int(row["n"])

    # -- the filter compiler ------------------------------------------------ #

    def compile(self, where: Filter | None, *, start: int = 1) -> _Sql:
        """Compile our filter algebra into a parameterised SQL predicate.

        Args:
            where: The filter.
            start: The next free placeholder number. The caller may already have used
                ``$1`` for the query vector.

        Every value — including every **field name** — is a bind parameter. Postgres will
        parameterise a JSON path (``metadata->$1``) even though it will not parameterise
        an identifier, which is what keeps a metadata key called
        ``'; DROP TABLE users; --`` from being anything other than a key that matches
        nothing.
        """
        if where is None:
            return _Sql("TRUE")

        params: list[Any] = []
        text = self._node(where, params, start)
        return _Sql(text, params)

    def _node(self, where: Filter, params: list[Any], start: int) -> str:
        """One node of the tree, appending its bind values to ``params``."""
        match where.kind:
            case "and":
                parts = [self._node(f, params, start) for f in where.filters]
                return "(" + " AND ".join(parts) + ")"
            case "or":
                parts = [self._node(f, params, start) for f in where.filters]
                return "(" + " OR ".join(parts) + ")"
            case "not":
                return f"(NOT {self._node(where.filter, params, start)})"
            case _:
                return self._comparison(where, params, start)

    def _comparison(self, where: Compare, params: list[Any], start: int) -> str:
        """One comparison, guarded so that it is never NULL.

        The guard is the interesting part. ``metadata->>'tenant' <> 'acme'`` on a row
        with no ``tenant`` evaluates to ``NULL``, and ``NOT NULL`` is ``NULL`` — so a
        negated filter would silently fail to match rows that are missing the field,
        which is not what the algebra says. ``metadata ? 'tenant' AND ...`` forces a real
        ``FALSE``, and ``NOT FALSE`` is ``TRUE``, which is.
        """

        def bind(value: Any) -> str:
            params.append(value)
            return f"${start + len(params) - 1}"

        key = bind(where.field)
        has_key = f"metadata ? {key}"

        if where.op is CompareOp.EXISTS:
            return has_key if where.value else f"(NOT {has_key})"

        match where.op:
            case CompareOp.EQ:
                return f"({has_key} AND metadata -> {key} = {bind(json.dumps(where.value))}::jsonb)"
            case CompareOp.NE:
                return (
                    f"({has_key} AND metadata -> {key} <> {bind(json.dumps(where.value))}::jsonb)"
                )
            case CompareOp.IN:
                return (
                    f"({has_key} AND metadata -> {key} IN "
                    f"(SELECT jsonb_array_elements({bind(json.dumps(where.value))}::jsonb)))"
                )
            case CompareOp.NOT_IN:
                return (
                    f"({has_key} AND metadata -> {key} NOT IN "
                    f"(SELECT jsonb_array_elements({bind(json.dumps(where.value))}::jsonb)))"
                )
            case _:
                return self._ordering(where, has_key, key, bind)

    @staticmethod
    def _ordering(where: Compare, has_key: str, key: str, bind: Any) -> str:
        """``>``, ``>=``, ``<``, ``<=`` — the ones whose SQL depends on the value's type.

        ``metadata->>'year'`` is *text*, and text orders lexically: ``'9' > '10'``. So a
        numeric comparison has to cast, and a string comparison must not. Unlike Qdrant,
        Postgres can order both, so both are supported rather than one being refused.
        """
        symbol = {
            CompareOp.GT: ">",
            CompareOp.GTE: ">=",
            CompareOp.LT: "<",
            CompareOp.LTE: "<=",
        }[where.op]

        numeric = isinstance(where.value, int | float) and not isinstance(where.value, bool)
        lhs = f"(metadata ->> {key})::numeric" if numeric else f"metadata ->> {key}"
        rhs = bind(where.value) if numeric else bind(str(where.value))

        return f"({has_key} AND {lhs} {symbol} {rhs})"

    # -- translation back --------------------------------------------------- #

    def _hit(self, row: Any, distance: Distance, include_vectors: bool) -> SearchHit:
        """One row, with its score pointing the way the port promises."""
        return SearchHit(
            id=row["id"],
            score=_score(float(row["distance"]), distance),
            metadata=_parse_metadata(row["metadata"]),
            vector=_parse_vector(row["vector"]) if include_vectors else None,
        )

    def _fail(self, exc: Exception, collection: str) -> SarbError:
        """Translate a Postgres failure into one of ours."""
        if isinstance(exc, SarbError):
            return exc

        return VectorStoreError(
            f"Postgres call failed: {exc}",
            provider=PROVIDER,
            context={"collection": collection},
            cause=exc,
            retryable=True,
        )

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._pool.close()


# --------------------------------------------------------------------------- #
# Values on and off the wire
# --------------------------------------------------------------------------- #


def _score(value: float, distance: Distance) -> float:
    """Turn an operator's number into a score where higher is better.

    ``<=>`` gives cosine *distance* (0 identical, 2 opposite), ``<->`` gives Euclidean
    distance, and ``<#>`` gives the **negative** inner product — a quirk of pgvector,
    which only has ascending index scans and so negates the one metric where bigger is
    better. All three would rank backwards if handed straight to a caller.
    """
    match distance:
        case Distance.COSINE:
            return 1.0 - value
        case Distance.DOT:
            return -value
        case _:
            return 1.0 / (1.0 + value)


def _vector_literal(vector: Sequence[float]) -> str:
    """Pgvector's text form: ``[1,2,3]``.

    Sent as text and cast in the statement, so the adapter does not depend on the
    ``pgvector.asyncpg`` codec having been registered on the connection — one less thing
    that can be forgotten in a pool the application supplied.
    """
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def _parse_vector(raw: Any) -> list[float]:
    """The text form, back to floats."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(v) for v in raw]
    return [float(part) for part in str(raw).strip("[]").split(",") if part]


def _parse_metadata(raw: Any) -> dict[str, Any]:
    """Asyncpg hands back jsonb as a string unless a codec says otherwise."""
    if isinstance(raw, str):
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    return dict(raw or {})


async def build(ctx: ComponentContext) -> PgVectorStore:
    """Build the ``pgvector`` store, and its connection pool, from configuration.

    The only **async** factory in the framework, and the reason is not incidental: a
    connection pool opens sockets, and sockets need an event loop. ``asyncpg.create_pool``
    returns a pool that is not usable until it has been awaited, so a synchronous factory
    could only hand back something that fails on first use — which would move the
    failure from startup, where it belongs, to the first search a user runs.

    The container awaits a factory that returns an awaitable, so this costs nothing but
    the keyword.
    """
    asyncpg = import_optional("asyncpg", extra=EXTRA, used_by=f"vector store {PROVIDER!r}")
    options = parse_options(ctx.config, PgVectorOptions, component=ctx.path)

    pool = await asyncpg.create_pool(
        dsn=options.dsn.get_secret_value(),
        min_size=options.min_size,
        max_size=options.max_size,
    )
    return PgVectorStore(pool, options)
