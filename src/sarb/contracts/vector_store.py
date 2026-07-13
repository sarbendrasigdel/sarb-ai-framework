"""The vector-store port, and the neutral filter language it needs.

Filters are the hard part
-------------------------
Similarity search is easy to abstract: every store takes a vector and a ``k``.
*Metadata filtering* is where vendor neutrality actually gets decided, and it is
where most "pluggable vector store" abstractions quietly give up — they expose a
``dict`` that is passed through to the backend, and the moment an application
writes one, it is married to Qdrant forever.

So the framework defines a small filter algebra as data, and each adapter compiles
it to its backend. The algebra is deliberately *small*: equality, ordering, set
membership, existence, and boolean composition. That covers the overwhelming
majority of real filters, and everything in it can be expressed in Qdrant's filter
DSL and in SQL alike.

An adapter that cannot express a filter must raise
:class:`~sarb.errors.VectorStoreError` rather than silently ignore it. A dropped
filter returns *more* results, not fewer — so it fails open, quietly, and leaks
data across whatever boundary the filter was enforcing.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from sarb.contracts.base import JSONValue, Metadata, SarbModel
from sarb.contracts.component import Family
from sarb.contracts.embedding import Vector

__all__ = [
    "VECTOR_STORES",
    "And",
    "Compare",
    "CompareOp",
    "Distance",
    "Filter",
    "Not",
    "Or",
    "SearchHit",
    "VectorQuery",
    "VectorRecord",
    "VectorStore",
    "eq",
    "exists",
    "gt",
    "gte",
    "in_",
    "lt",
    "lte",
    "ne",
    "not_in",
]


class Distance(StrEnum):
    """How similarity is measured. Fixed when a collection is created."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


class CompareOp(StrEnum):
    """The comparisons every backend can express."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"


class _FilterOps(SarbModel):
    """Boolean composition, so filters read like the condition they express.

    ``eq("tenant", t) & gte("year", 2020)`` rather than
    ``And(filters=(Compare(...), Compare(...)))``. The verbose form still exists and
    is what gets serialised; this is only about the call site.
    """

    def __and__(self, other: Filter) -> And:
        """Both must hold."""
        return And(filters=(self, other))

    def __or__(self, other: Filter) -> Or:
        """Either may hold."""
        return Or(filters=(self, other))

    def __invert__(self) -> Not:
        """Must not hold."""
        return Not(filter=self)


class Compare(_FilterOps):
    """One comparison against a metadata field."""

    kind: Literal["compare"] = "compare"
    field: str
    op: CompareOp
    value: JSONValue = None

    @model_validator(mode="after")
    def _value_matches_op(self) -> Self:
        """Set operators need a list; EXISTS needs a bool. Caught here, not in SQL."""
        if self.op in (CompareOp.IN, CompareOp.NOT_IN) and not isinstance(self.value, list):
            raise ValueError(f"{self.op.value} needs a list value")
        if self.op is CompareOp.EXISTS and not isinstance(self.value, bool):
            raise ValueError("exists needs a bool value")
        return self


class And(_FilterOps):
    """Every sub-filter must hold."""

    kind: Literal["and"] = "and"
    filters: tuple[Filter, ...]


class Or(_FilterOps):
    """At least one sub-filter must hold."""

    kind: Literal["or"] = "or"
    filters: tuple[Filter, ...]


class Not(_FilterOps):
    """The sub-filter must not hold."""

    kind: Literal["not"] = "not"
    filter: Filter


#: A filter expression. Recursive and discriminated on ``kind``, so it round-trips
#: through JSON and can be stored, logged, or sent over a wire.
type Filter = Annotated[Compare | And | Or | Not, Field(discriminator="kind")]


# -- constructors ----------------------------------------------------------- #


def eq(field: str, value: JSONValue) -> Compare:
    """``field == value``."""
    return Compare(field=field, op=CompareOp.EQ, value=value)


def ne(field: str, value: JSONValue) -> Compare:
    """``field != value``."""
    return Compare(field=field, op=CompareOp.NE, value=value)


def gt(field: str, value: JSONValue) -> Compare:
    """``field > value``."""
    return Compare(field=field, op=CompareOp.GT, value=value)


def gte(field: str, value: JSONValue) -> Compare:
    """``field >= value``."""
    return Compare(field=field, op=CompareOp.GTE, value=value)


def lt(field: str, value: JSONValue) -> Compare:
    """``field < value``."""
    return Compare(field=field, op=CompareOp.LT, value=value)


def lte(field: str, value: JSONValue) -> Compare:
    """``field <= value``."""
    return Compare(field=field, op=CompareOp.LTE, value=value)


def in_(field: str, values: list[JSONValue]) -> Compare:
    """``field`` is one of ``values``."""
    return Compare(field=field, op=CompareOp.IN, value=values)


def not_in(field: str, values: list[JSONValue]) -> Compare:
    """``field`` is none of ``values``."""
    return Compare(field=field, op=CompareOp.NOT_IN, value=values)


def exists(field: str, *, present: bool = True) -> Compare:
    """``field`` is (or is not) present."""
    return Compare(field=field, op=CompareOp.EXISTS, value=present)


# --------------------------------------------------------------------------- #
# Records and queries
# --------------------------------------------------------------------------- #


class VectorRecord(SarbModel):
    """One stored vector and the metadata attached to it.

    Attributes:
        id: Caller-chosen and stable. Upserting the same id replaces the record,
            which is what makes re-indexing a changed document idempotent rather
            than duplicating it.
        vector: The embedding.
        metadata: What filters match on, and what search returns alongside the hit.
    """

    id: str
    vector: Vector
    metadata: Metadata = Field(default_factory=dict)


class VectorQuery(SarbModel):
    """A similarity search.

    Attributes:
        vector: What to search near.
        top_k: How many hits to return.
        where: Metadata filter. Applied *with* the similarity search, not after —
            filtering post-hoc would silently return fewer than ``top_k``.
        min_score: Drop hits below this similarity.
        include_vectors: Return the stored vectors too. Off by default: they are
            large, and almost nobody wants them.
    """

    vector: Vector
    top_k: int = Field(default=10, gt=0, le=1000)
    where: Filter | None = None
    min_score: float | None = None
    include_vectors: bool = False


class SearchHit(SarbModel):
    """One result.

    Attributes:
        id: The record's id.
        score: Similarity, **higher is always better** — regardless of the
            collection's distance metric. Euclidean distance is a *distance*, so an
            adapter using it must invert it before reporting. Normalising the
            direction here means retrieval code can sort one way, always, without
            asking which metric the store happens to use.
        metadata: The record's metadata.
        vector: The stored vector, if it was asked for.
    """

    id: str
    score: float
    metadata: Metadata = Field(default_factory=dict)
    vector: Vector | None = None


# --------------------------------------------------------------------------- #
# Port
# --------------------------------------------------------------------------- #


@runtime_checkable
class VectorStore(Protocol):
    """Stores vectors and searches them."""

    async def ensure_collection(
        self,
        collection: str,
        *,
        dimensions: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Create the collection if it is absent.

        Idempotent, so it can be called on every boot. If the collection exists with
        *different* dimensions or distance, raise rather than adapt: silently
        writing 768-dimension vectors into a 1536-dimension collection, or changing
        the metric under an existing index, corrupts retrieval in a way that shows
        up as "the answers got worse" weeks later.

        Raises:
            VectorStoreError: The collection exists with an incompatible shape.
        """
        ...

    async def drop_collection(self, collection: str) -> None:
        """Delete the collection and everything in it. Idempotent."""
        ...

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        """Insert or replace records by id.

        Batch-first, for the same reason as embedding: the per-call overhead
        dominates.

        Raises:
            VectorStoreError: The write failed, or a vector's width does not match
                the collection's.
        """
        ...

    async def search(self, collection: str, query: VectorQuery) -> Sequence[SearchHit]:
        """Find the nearest records.

        Returns:
            Hits, best first, at most ``top_k``.

        Raises:
            VectorStoreError: The search failed, or the filter uses something this
                backend cannot express. Never silently drop a filter — a dropped
                filter returns *more* results, so it fails open.
        """
        ...

    async def fetch(self, collection: str, ids: Sequence[str]) -> Sequence[VectorRecord]:
        """Get records by id. Missing ids are omitted, not raised."""
        ...

    async def delete(
        self,
        collection: str,
        *,
        ids: Sequence[str] | None = None,
        where: Filter | None = None,
    ) -> int:
        """Delete by id or by filter.

        Exactly one of ``ids`` or ``where``. Requiring one of them is deliberate:
        ``delete(collection)`` with no argument is one slip away from erasing a
        production index, so there is no such call to slip into.

        Returns:
            How many records were deleted.

        Raises:
            InvalidInputError: Neither or both of ``ids`` and ``where`` were given.
        """
        ...

    async def count(self, collection: str, *, where: Filter | None = None) -> int:
        """How many records match."""
        ...


#: The ``vector_stores`` config family.
VECTOR_STORES = Family("vector_stores", VectorStore)
