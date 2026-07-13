"""An LLM provider with a guard around it.

Twenty lines, and none of them are about retrying or tracing — that is the point of
the split. This class knows what an :class:`~sarb.contracts.LLMRequest` is and where
the token counts live; the :class:`~sarb.middleware.guard.Guard` knows what to do when
a call fails. Neither knows what the other knows.

It is an ``LLMProvider`` and it wraps an ``LLMProvider``, which — because the port is a
``Protocol`` — means it works on an adapter that does not exist yet, including one from
a plugin that never imported this framework.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sarb.contracts import (
    LLMCapabilities,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    StreamEnd,
    StreamEvent,
)
from sarb.middleware import semconv
from sarb.middleware.guard import Guard, Outcome

__all__ = ["GuardedLLM"]

OPERATION = "chat"


class GuardedLLM:
    """The :class:`~sarb.contracts.LLMProvider` port, retried and instrumented."""

    def __init__(self, inner: LLMProvider, guard: Guard) -> None:
        self._inner = inner
        self._guard = guard

    @property
    def model(self) -> str:
        """The model the wrapped provider was configured with."""
        return self._inner.model

    @property
    def capabilities(self) -> LLMCapabilities:
        """The wrapped provider's capabilities. A guard adds none and removes none."""
        return self._inner.capabilities

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate, retrying a retryable failure and recording what happened.

        Safe to retry, and that is not an accident of the implementation: ``generate``
        is a pure function of its request as far as the caller is concerned. An agent
        loop that *executes tools* between turns is a different matter — it has side
        effects, it is not idempotent, and it will get its own policy rather than
        inheriting this one.
        """
        return await self._guard.run(
            OPERATION,
            lambda: self._inner.generate(request),
            target=self._inner.model,
            # The guard does not know that its target is a *model*. This wrapper does,
            # so this wrapper is what says so — and what keeps a collection name out of
            # the field every GenAI dashboard reads as a model.
            attributes={semconv.REQUEST_MODEL: self._inner.model},
            outcome=_outcome,
        )

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Stream, retrying only until the first event has been handed over.

        See :meth:`~sarb.middleware.guard.Guard.stream` for why that line is where it
        is: once a token has reached the caller it has been rendered, and no retry can
        take it back.

        The tokens are recorded from :class:`~sarb.contracts.StreamEnd`, which is the
        entire reason that event carries a fully assembled response. Without this hook a
        streamed call reported no usage and no cost whatsoever — and since anything with
        a chat interface streams, that was most of the spend, missing.
        """
        return self._guard.stream(
            OPERATION,
            lambda: self._inner.stream(request),
            target=self._inner.model,
            attributes={semconv.REQUEST_MODEL: self._inner.model},
            outcome=_stream_outcome,
        )

    async def aclose(self) -> None:
        """Close the wrapped provider.

        The container closes what it built, and what it built is this wrapper. Without
        this line the adapter underneath would keep its connection pool open until the
        process died.
        """
        closer = getattr(self._inner, "aclose", None)
        if closer is not None:
            await closer()


def _outcome(response: LLMResponse) -> Outcome:
    """Tell the guard what a finished chat call is worth recording."""
    return Outcome(
        model=response.model,
        usage=response.usage,
        finish_reason=response.finish_reason.value,
    )


def _stream_outcome(event: StreamEvent) -> Outcome | None:
    """The same, from the one event in a stream that knows how the call went.

    Every other event is a fragment — a few characters of text, part of a tool call's
    JSON — and knows nothing about totals. ``None`` for those, and the guard keeps
    looking.
    """
    if isinstance(event, StreamEnd):
        return _outcome(event.response)
    return None
