"""The embedding port.

The one non-obvious thing here is :class:`EmbeddingKind`, and it is the difference
between a RAG system that works and one that quietly retrieves the wrong things.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import Field

from sarb.contracts.base import SarbModel
from sarb.contracts.component import Family
from sarb.contracts.message import Usage

__all__ = [
    "EMBEDDERS",
    "EmbeddingKind",
    "EmbeddingProvider",
    "EmbeddingResponse",
    "Vector",
]

#: A dense embedding. A plain list of floats: the contracts ring depends on nothing
#: but Pydantic and the standard library, so no numpy array here. Adapters and
#: vector stores are free to use numpy internally, and do.
type Vector = list[float]


class EmbeddingKind(StrEnum):
    """Whether text is being embedded as a query or as a document.

    Many strong retrieval models — BGE, E5, and most instruction-tuned embedders —
    are **asymmetric**: they expect a different prefix for the thing being searched
    for than for the thing being searched. Embed a query as a document and you get
    a vector that lands in slightly the wrong place. Nothing errors. Recall just
    quietly drops, and the system looks like it works.

    Making the caller state which one it means is the only way to keep that bug out
    of a framework whose whole point is that the embedding model is swappable.
    Symmetric models simply ignore it.
    """

    DOCUMENT = "document"
    """Text being indexed."""

    QUERY = "query"
    """Text being searched with."""


class EmbeddingResponse(SarbModel):
    """The vectors, and what they cost.

    ``embed()`` used to return a bare ``list[Vector]``, which was pleasant to call and
    had one flaw: there was nowhere to put the token count. So embedding was the one
    priced operation in the framework that no dashboard could see, and a full re-index
    of a corpus — the single largest embedding bill anyone ever gets — was invisible.

    An extra attribute access at every call site is the price of that number existing.
    It is also what :class:`~sarb.contracts.LLMProvider` has always done: ``generate()``
    returns a response object carrying usage, and this port returning a naked list was
    the inconsistency, not the convenience.

    Attributes:
        vectors: One vector per input text, in the same order.
        model: The model that actually produced them.
        usage: Tokens consumed. Zero for a local model, which has no bill to speak of —
            see :class:`~sarb.contracts.Usage` for what the fields mean.
    """

    vectors: list[Vector]
    model: str
    usage: Usage = Field(default_factory=Usage)

    def __len__(self) -> int:
        """How many vectors came back. Lets `len(response)` mean the obvious thing."""
        return len(self.vectors)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors."""

    @property
    def model(self) -> str:
        """The embedding model in use. Changing it invalidates every stored vector."""
        ...

    @property
    def dimensions(self) -> int:
        """The vector width. Must match the collection the vectors are written to."""
        ...

    async def embed(
        self,
        texts: Sequence[str],
        *,
        kind: EmbeddingKind = EmbeddingKind.DOCUMENT,
    ) -> EmbeddingResponse:
        """Embed a batch of texts.

        Batch-first: the signature takes a sequence, not a string. Embedding is
        dominated by per-call overhead — an HTTP round trip, or a model forward
        pass — so a one-at-a-time API invites callers to write the slow loop, and
        they do. Embedding one string is ``(await embed([text])).vectors[0]``.

        Args:
            texts: The texts to embed.
            kind: Whether these are documents or queries. See :class:`EmbeddingKind`.

        Returns:
            An :class:`EmbeddingResponse` whose ``vectors`` hold one vector per input,
            in the same order — always the same length as ``texts``.

        Raises:
            EmbeddingError: The call failed.
        """
        ...


#: The ``embedders`` config family.
EMBEDDERS = Family("embedders", EmbeddingProvider)
