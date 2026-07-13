"""An embedding provider with a guard around it.

Twelve lines. The whole reason the :class:`~sarb.middleware.guard.Guard` was factored
out of the LLM wrapper is visible here: this file contains no retry logic, no backoff
maths, no span, and no metric — and it gets all four.
"""

from __future__ import annotations

from collections.abc import Sequence

from sarb.contracts import EmbeddingKind, EmbeddingProvider, EmbeddingResponse
from sarb.middleware import semconv
from sarb.middleware.guard import Guard, Outcome

__all__ = ["GuardedEmbedder"]

OPERATION = "embeddings"


class GuardedEmbedder:
    """The :class:`~sarb.contracts.EmbeddingProvider` port, retried and instrumented."""

    def __init__(self, inner: EmbeddingProvider, guard: Guard) -> None:
        self._inner = inner
        self._guard = guard

    @property
    def model(self) -> str:
        """The wrapped provider's model."""
        return self._inner.model

    @property
    def dimensions(self) -> int:
        """The wrapped provider's vector width."""
        return self._inner.dimensions

    async def embed(
        self,
        texts: Sequence[str],
        *,
        kind: EmbeddingKind = EmbeddingKind.DOCUMENT,
    ) -> EmbeddingResponse:
        """Embed, retrying a retryable failure and recording what happened.

        Idempotent, so retrying is safe. It is also where retrying *matters* most:
        indexing is the operation that fires hundreds of requests in a burst, and so
        it is the one that gets rate limited. A rate limit that reaches the caller here
        means a half-indexed corpus, and half an index is worse than none, because it
        looks like one.
        """
        return await self._guard.run(
            OPERATION,
            lambda: self._inner.embed(texts, kind=kind),
            target=self._inner.model,
            attributes={semconv.REQUEST_MODEL: self._inner.model},
            outcome=_outcome,
        )

    async def aclose(self) -> None:
        """Close the wrapped provider, if it holds anything worth closing."""
        closer = getattr(self._inner, "aclose", None)
        if closer is not None:
            await closer()


def _outcome(response: EmbeddingResponse) -> Outcome:
    """Tell the guard what a finished embedding call is worth recording.

    A local embedder reports no tokens, and the guard records no zero for them — see
    :meth:`~sarb.middleware.guard.Guard._record_tokens`. So this costs nothing when
    there is nothing to cost.
    """
    return Outcome(model=response.model, usage=response.usage)
