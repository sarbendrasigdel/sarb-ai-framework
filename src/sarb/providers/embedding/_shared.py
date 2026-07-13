"""Machinery every embedding adapter needs.

Embedding looks like the easy port — text in, floats out — and that is exactly why
it is worth being careful here. Its failure mode is not an exception. It is a vector
that is the wrong width, or in the wrong position, written happily into an index that
will not complain for six months, at which point retrieval is merely *worse* and
nobody can say when it started.

So this module owns the three things that keep that from happening: the promise that
output lines up with input, the batching that keeps a large call from being rejected,
and the thread discipline that keeps a local model off the event loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar

import anyio
import anyio.to_thread
from pydantic import BaseModel, ConfigDict, Field

from sarb.contracts import Vector
from sarb.errors import EmbeddingError

__all__ = [
    "EmbeddingOptions",
    "LocalEmbeddingOptions",
    "ThreadRunner",
    "batches",
    "check_output",
    "normalized",
]

T = TypeVar("T")


class EmbeddingOptions(BaseModel):
    """The options every embedding adapter accepts, whatever it wraps.

    Attributes:
        model: The embedding model. Required, and not defaulted — changing it
            invalidates every vector already stored, so it is not a value anything
            should choose on your behalf.
        batch_size: How many texts go to the model at once. The adapter splits
            larger calls up itself, because a caller with ten thousand chunks
            should get ten thousand vectors, not a ``400``.
        normalize: Scale every vector to unit length.

    Why ``normalize`` is here and not left to the vector store:
        Cosine similarity over un-normalised vectors, in an index configured for dot
        product, does not fail. It just ranks badly. The models disagree about
        whether they normalise for you — most sentence-transformers do not, most
        fastembed models do — so the framework has to let you say. Normalising a
        vector that is already unit length is a no-op, which is why this can be
        applied uniformly and safely.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    model: str = Field(min_length=1)
    batch_size: int = Field(default=64, gt=0)
    normalize: bool = False


class LocalEmbeddingOptions(EmbeddingOptions):
    """Options shared by adapters that run the model in this process.

    Attributes:
        cache_dir: Where downloaded weights live. Set it in a container, or the
            model is re-downloaded on every cold start.
        max_concurrency: How many embedding calls may be in the model at once.

    Why ``max_concurrency`` defaults to 1:
        A local model is one object holding one set of weights, and a forward pass is
        CPU- or GPU-bound rather than IO-bound. Running fifty of them at once does not
        make any of them faster; it multiplies peak memory by fifty and takes the
        process down. Queuing is the correct behaviour, and it is not the default that
        ``to_thread`` would give you, so it is stated.
    """

    cache_dir: str | None = None
    max_concurrency: int = Field(default=1, gt=0)


class ThreadRunner:
    """Runs a blocking model call off the event loop, and not too many at once.

    The port is async; a local model is not. Calling it directly would stall the
    entire process — every other request, every health check, every open stream —
    for as long as the forward pass takes, which for a batch of documents is not
    milliseconds.

    Handing it to a thread fixes that and introduces the opposite problem, so the
    limiter is not optional. See :attr:`LocalEmbeddingOptions.max_concurrency`.
    """

    def __init__(self, max_concurrency: int) -> None:
        self._limiter = anyio.CapacityLimiter(max_concurrency)

    async def run(self, work: Callable[[], T]) -> T:
        """Run ``work`` in a worker thread, waiting for a slot if they are all busy."""
        return await anyio.to_thread.run_sync(work, limiter=self._limiter)


def batches(texts: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    """Split a call into slices the model will accept.

    Callers embed a document, which is however many chunks the document turned out to
    be. Hosted APIs cap the number of inputs per request, and local models cap it with
    memory. Neither limit is the caller's business — the port promises to embed what it
    is given — so the adapter does the splitting.
    """
    for start in range(0, len(texts), size):
        yield texts[start : start + size]


def normalized(vector: Vector) -> Vector:
    """Scale a vector to unit length, tolerating the zero vector.

    A zero vector has no direction to preserve, so it is returned unchanged rather
    than turned into a division by zero. It is a legitimate thing for a model to emit
    for empty or unusual input, and it is not worth an exception.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def check_output(
    vectors: list[Vector],
    texts: Sequence[str],
    *,
    provider: str,
    model: str,
    dimensions: int,
) -> list[Vector]:
    """Assert the two promises the port makes, before anything can rely on them.

    One vector per input, in the same order, all of the expected width.

    This looks like paranoia and is not. Both failures are **silent**: a vector store
    accepts a misaligned batch without complaint, and every chunk in it is then
    findable only by the text of some *other* chunk. Nothing raises. Nothing is
    logged. Recall is simply worse than it should be, for as long as the index lives.

    The width check earns its keep on the day someone changes ``model`` in YAML and
    not the collection behind it — or when a hosted API quietly ignores the
    ``dimensions`` it was sent, which they do.

    Raises:
        EmbeddingError: The provider returned the wrong number of vectors, or a
            vector of the wrong width.
    """
    context = {"model": model, "expected": len(texts), "received": len(vectors)}

    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"provider returned {len(vectors)} vectors for {len(texts)} texts",
            provider=provider,
            context=context,
        )

    for index, vector in enumerate(vectors):
        if len(vector) != dimensions:
            raise EmbeddingError(
                f"provider returned a {len(vector)}-dimensional vector where "
                f"{dimensions} was configured",
                provider=provider,
                context={"model": model, "index": index, "dimensions": len(vector)},
            )

    return vectors
