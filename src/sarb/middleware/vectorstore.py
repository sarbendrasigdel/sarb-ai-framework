"""A vector store with a guard around it.

The third port to get one, and the first where the *write* path is worth as much as the
read path: a rate limit or a dropped connection halfway through indexing a corpus leaves
a partial index, and a partial index is worse than no index, because it looks like one.

Every method here is idempotent, which is what makes retrying them safe rather than
merely convenient. ``upsert`` is keyed by id and replaces; ``delete`` of an already-gone
record removes nothing and says so; ``ensure_collection`` is defined as idempotent by
the port. Retry was designed into this port before it had an implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

from sarb.contracts import (
    Distance,
    Filter,
    SearchHit,
    VectorQuery,
    VectorRecord,
    VectorStore,
)
from sarb.middleware.guard import Guard

__all__ = ["GuardedVectorStore"]


class GuardedVectorStore:
    """The :class:`~sarb.contracts.VectorStore` port, retried and instrumented."""

    def __init__(self, inner: VectorStore, guard: Guard) -> None:
        self._inner = inner
        self._guard = guard

    async def ensure_collection(
        self,
        collection: str,
        *,
        dimensions: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Create the collection if absent. Idempotent, so safe to retry."""
        await self._guard.run(
            "ensure_collection",
            lambda: self._inner.ensure_collection(
                collection, dimensions=dimensions, distance=distance
            ),
            target=collection,
        )

    async def drop_collection(self, collection: str) -> None:
        """Delete the collection and everything in it."""
        await self._guard.run(
            "drop_collection",
            lambda: self._inner.drop_collection(collection),
            target=collection,
        )

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        """Insert or replace by id. Retried, because a half-written index is a trap."""
        await self._guard.run(
            "upsert",
            lambda: self._inner.upsert(collection, records),
            target=collection,
        )

    async def search(self, collection: str, query: VectorQuery) -> Sequence[SearchHit]:
        """Search."""
        return await self._guard.run(
            "search",
            lambda: self._inner.search(collection, query),
            target=collection,
        )

    async def fetch(self, collection: str, ids: Sequence[str]) -> Sequence[VectorRecord]:
        """Get by id."""
        return await self._guard.run(
            "fetch",
            lambda: self._inner.fetch(collection, ids),
            target=collection,
        )

    async def delete(
        self,
        collection: str,
        *,
        ids: Sequence[str] | None = None,
        where: Filter | None = None,
    ) -> int:
        """Delete by id or by filter."""
        return await self._guard.run(
            "delete",
            lambda: self._inner.delete(collection, ids=ids, where=where),
            target=collection,
        )

    async def count(self, collection: str, *, where: Filter | None = None) -> int:
        """How many records match."""
        return await self._guard.run(
            "count",
            lambda: self._inner.count(collection, where=where),
            target=collection,
        )

    async def aclose(self) -> None:
        """Close the wrapped store, if it holds anything worth closing."""
        closer = getattr(self._inner, "aclose", None)
        if closer is not None:
            await closer()
