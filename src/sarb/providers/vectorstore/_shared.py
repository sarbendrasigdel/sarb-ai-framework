"""Machinery every vector-store adapter needs.

The theme of this module is that a vector store fails *open*. A dropped filter returns
more rows, not fewer. A mis-sized vector lands in an index that will happily hold it. A
score sorted the wrong way round produces results — just the wrong ones. None of these
raise anywhere unless somebody makes them.

So the checks that make them raise live here, once, and every adapter goes through them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from sarb.contracts import Distance, VectorRecord
from sarb.errors import InvalidInputError, VectorStoreError

__all__ = [
    "VectorStoreOptions",
    "check_delete_selector",
    "check_dimensions",
    "score_from_distance",
]


class VectorStoreOptions(BaseModel):
    """The options every vector-store adapter accepts.

    Attributes:
        collection: The default collection, when a caller does not name one.
        batch_size: How many records go to the backend in one write. The adapter
            splits larger writes itself — a caller indexing a book should not have to
            know the backend's request size limit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    collection: str = Field(default="documents", min_length=1)
    batch_size: int = Field(default=128, gt=0)


def check_delete_selector(
    ids: Sequence[str] | None,
    where: object | None,
    *,
    provider: str,
) -> None:
    """Exactly one of ``ids`` or ``where``, and the port means exactly.

    ``delete(collection)`` with neither is one slip away from erasing a production
    index, so there is no such call to slip into. Passing both is not a convenience
    either — it is ambiguous (union or intersection?), and an adapter that guessed
    would guess differently from the next one.

    Raises:
        InvalidInputError: Neither or both were given.
    """
    if (ids is None) == (where is None):
        raise InvalidInputError(
            "delete needs exactly one of `ids` or `where`",
            context={"provider": provider, "ids": ids is not None, "where": where is not None},
        )


def check_dimensions(
    records: Sequence[VectorRecord],
    *,
    dimensions: int,
    collection: str,
    provider: str,
) -> None:
    """Every vector must be as wide as the collection it is going into.

    Both backends would reject this themselves, eventually, with a message about their
    own internals. Catching it here means the error names the record, before a network
    round trip, and says the thing that is actually wrong: the embedder and the
    collection disagree, which almost always means someone changed ``model:`` in the
    ``embedders`` block and did not rebuild the index.

    Raises:
        VectorStoreError: A vector is the wrong width.
    """
    for record in records:
        if len(record.vector) != dimensions:
            raise VectorStoreError(
                f"record {record.id!r} has a {len(record.vector)}-dimensional vector, but "
                f"collection {collection!r} holds {dimensions}-dimensional ones",
                provider=provider,
                context={
                    "collection": collection,
                    "id": record.id,
                    "expected": dimensions,
                    "received": len(record.vector),
                },
            )


def score_from_distance(value: float, distance: Distance) -> float:
    """Normalise a backend's number so that **higher is always better**.

    Cosine and dot product already report similarity, so they pass through untouched.
    Euclidean reports a *distance* — smaller is closer — and handing that back as a
    "score" would invert every ranking in the system while raising nothing at all. The
    search would work. It would return the furthest documents first, and the only
    symptom would be that the answers are bad.

    So Euclidean is mapped through ``1 / (1 + d)``: monotonic, so the ordering is exactly
    the backend's, and bounded to ``(0, 1]``, so ``min_score`` has a scale to mean
    something on.

    Note:
        The *absolute* value is not comparable across metrics, and no transform could
        make it so. A ``min_score`` tuned against cosine means something different
        against Euclidean, exactly as it means something different against dot product.
        That is a property of the metrics, not of this function.
    """
    if distance is Distance.EUCLIDEAN:
        return 1.0 / (1.0 + value)
    return value


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. For the in-process store; the real backends do their own."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
