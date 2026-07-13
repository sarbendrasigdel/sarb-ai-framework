"""Narrowing helpers shared by the adapter tests.

The stream is a discriminated union, so reading ``.response`` off the last event
means proving it is a :class:`~sarb.contracts.StreamEnd` first. That is not
ceremony — it is the type system enforcing the same thing the port promises, and
these two helpers do it once instead of in forty assertions.
"""

from __future__ import annotations

from collections.abc import Sequence

from sarb.contracts import LLMResponse, ReasoningPart, StreamEnd, StreamEvent


def end_of(events: Sequence[StreamEvent]) -> LLMResponse:
    """The assembled response a stream must end with.

    Asserts the promise the port makes: the final event of every stream is a
    :class:`~sarb.contracts.StreamEnd` carrying the whole response — even when the
    stream produced no text at all.
    """
    end = events[-1]
    assert isinstance(end, StreamEnd), f"a stream must end with StreamEnd, got {type(end).__name__}"
    return end.response


def reasoning_of(response: LLMResponse) -> ReasoningPart:
    """The response's reasoning part, proven to be one."""
    part = response.message.content[0]
    assert isinstance(part, ReasoningPart), f"expected reasoning first, got {type(part).__name__}"
    return part
