"""The retrieval port.

A retrieval strategy is the whole "question in, passages out" step: embed, search,
maybe rerank, maybe query-expand, maybe hybrid-search, maybe all four. Everything
above it — an agent, a RAG chain, a summariser — depends only on this one method.

That is the point of drawing the boundary here rather than at the vector store.
Swapping naive top-k for hybrid search with a reranker is the single most common
upgrade a RAG system makes, and it should be a config change, not a rewrite of the
answering code:

    retrievers:
      default:
        provider: hybrid          # was: vector
        options:
          reranker: bge-reranker-v2
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import Field

from sarb.contracts.base import Metadata, SarbModel
from sarb.contracts.component import Family
from sarb.contracts.document import Chunk
from sarb.contracts.vector_store import Filter

__all__ = [
    "RETRIEVERS",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalStrategy",
]


class RetrievalQuery(SarbModel):
    """A request for relevant passages.

    Attributes:
        text: What the user asked. Text, not a vector: which embedding model to use
            is the *strategy's* business, and a query that arrived pre-embedded
            would have already made that choice on its behalf — and made hybrid or
            multi-vector strategies impossible to swap in.
        top_k: How many passages to return.
        where: Metadata filter. This is where tenant isolation lives, so a strategy
            must never drop it. See :mod:`sarb.contracts.vector_store`.
        min_score: Drop passages below this relevance.
        metadata: Strategy-specific extras.
    """

    text: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, le=100)
    where: Filter | None = None
    min_score: float | None = None
    metadata: Metadata = Field(default_factory=dict)


class RetrievalResult(SarbModel):
    """One retrieved passage.

    Attributes:
        chunk: The passage, with its provenance intact — which is what lets the
            answer be cited.
        score: Relevance, higher is better. Comparable *within* one result set only:
            a reranker's score and a cosine similarity are not the same quantity,
            and comparing them across strategies is meaningless.
        strategy: Which strategy produced it. Earns its place in a hybrid or
            ensemble retriever, where "why did this passage surface?" is otherwise
            unanswerable.
    """

    chunk: Chunk
    score: float
    strategy: str | None = None

    @property
    def text(self) -> str:
        """The passage text."""
        return self.chunk.text


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Finds passages relevant to a question."""

    async def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalResult]:
        """Retrieve relevant passages.

        Args:
            query: What to look for.

        Returns:
            Results, most relevant first, at most ``top_k``.

        Raises:
            RetrievalError: Retrieval failed.
        """
        ...


#: The ``retrievers`` config family.
RETRIEVERS = Family("retrievers", RetrievalStrategy)
