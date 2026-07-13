"""Machinery every LLM adapter needs, and none of them should own alone.

Four adapters have to do the same two awkward things: assemble a response out of
streamed fragments, and decide what a broken JSON argument string means. Left to
themselves they would each do it slightly differently — and the differences would
only surface in production, on the paths nobody exercises.

So those two things live here, once. The third — turning a vendor's HTTP status
into one of our errors — is not specific to language models, and lives one level up
in :mod:`sarb.providers._shared`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sarb.contracts import (
    ContentPart,
    FinishReason,
    LLMResponse,
    Message,
    ReasoningPart,
    Role,
    TextPart,
    ToolCall,
    Usage,
    new_id,
)

__all__ = [
    "LLMOptions",
    "ResponseBuilder",
    "parse_arguments",
]


class LLMOptions(BaseModel):
    """The options every LLM adapter accepts, whatever it wraps.

    Adapters subclass this and add their own. The base exists so that the knobs
    an *operator* thinks of as universal — which model, how long to wait, how many
    times to retry — are spelled the same way in every provider's YAML block.

    Attributes:
        model: The model to call. Required, and deliberately not defaulted: a
            framework that silently picks a model for you picks the wrong one, and
            bills you for it.
        base_url: Override the API endpoint. Points an adapter at a gateway, a
            proxy, or a self-hosted API-compatible server.
        timeout: Per-request deadline, in seconds.
        max_retries: Transport-level retries performed by the vendor SDK. This is
            *not* the framework's resilience policy; it is the SDK's own, and it
            is exposed so it can be turned off (``0``) when a higher layer owns
            retry.
        max_input_tokens: The model's context window, when the operator knows it.
        max_output_tokens: The model's output cap, when the operator knows it.

    Why the two token limits are configuration and not a lookup table:
        The obvious alternative is for the framework to ship a dict of model names
        to context windows. That table is wrong the week after it is written —
        vendors extend context windows, rename models and add aliases — and being
        wrong here is worse than being silent, because a budgeting engine would
        trust it. Same reasoning that keeps cost out of
        :class:`~sarb.contracts.Usage`: do not put a number in the domain model
        that someone else can change without telling you.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    model: str = Field(min_length=1)
    base_url: str | None = None
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_input_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)


# --------------------------------------------------------------------------- #
# Tool arguments
# --------------------------------------------------------------------------- #


def parse_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    """Parse a model's tool-call arguments, tolerating the model getting it wrong.

    Args:
        raw: The JSON string the model emitted. Empty or absent means "no
            arguments", which is different from "invalid arguments".

    Returns:
        ``(arguments, malformed)``. On success, the parsed dict and ``None``. On
        failure, an empty dict and the **raw string**, which the caller puts in
        :attr:`~sarb.contracts.ToolCall.malformed_arguments`.

    Never raises. That is the whole point: a model emitting broken JSON is a
    recoverable event, not a failed request — the tool engine hands the raw text
    back to the model, which is entirely capable of fixing it. Raising here would
    turn a retry into an outage; coercing to ``{}`` would invent an intent the
    model never had, and call a tool with no arguments as if that were what was
    asked for.

    A JSON value that parses but is not an object (``"[1, 2]"``, ``"null"``) is
    also malformed: tool arguments are named, always.
    """
    if raw is None or not raw.strip():
        return {}, None

    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}, raw

    if not isinstance(parsed, dict):
        return {}, raw

    return parsed, None


# --------------------------------------------------------------------------- #
# Assembling a response
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _PartialCall:
    """One tool call, possibly still arriving."""

    call_id: str | None = None
    name: str | None = None
    raw_arguments: str = ""
    arguments: dict[str, Any] | None = None


@dataclass(slots=True)
class ResponseBuilder:
    """Assembles one :class:`~sarb.contracts.LLMResponse` from fragments.

    Every adapter needs this, because every adapter has to honour the promise made
    by :class:`~sarb.contracts.StreamEnd`: that the last event of a stream carries
    the whole, assembled response. Without a shared implementation each adapter
    reassembles deltas its own way, and the ways differ — one forgets that a
    reasoning signature arrives *after* the reasoning it signs, another
    concatenates tool arguments in arrival order rather than by index.

    It is used by the non-streaming path too, so that both paths in one adapter
    produce a response of exactly the same shape. A provider whose ``generate``
    and ``stream`` disagree about where reasoning goes is a provider that fails
    only for the callers who stream.

    Example:
        >>> builder = ResponseBuilder("claude-opus-4-8")
        >>> builder.add_text("Par")
        >>> builder.add_text("is.")
        >>> builder.build().text
        'Paris.'
    """

    model: str
    _text: list[str] = field(default_factory=list)
    _reasoning: list[str] = field(default_factory=list)
    _signature: str | None = None
    _calls: dict[int, _PartialCall] = field(default_factory=dict)

    # -- accumulating ------------------------------------------------------- #

    def add_text(self, delta: str) -> None:
        """Append a fragment of the answer."""
        if delta:
            self._text.append(delta)

    def add_reasoning(self, delta: str, *, signature: str | None = None) -> None:
        """Append a fragment of exposed reasoning, and any signature with it."""
        if delta:
            self._reasoning.append(delta)
        if signature:
            self._signature = signature

    def set_signature(self, signature: str) -> None:
        """Record the reasoning signature, which providers send separately."""
        self._signature = signature

    def add_tool_call(self, *, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        """Record a complete tool call, already parsed.

        For providers that hand over whole calls rather than streaming their
        arguments as partial JSON.
        """
        self._calls[len(self._calls)] = _PartialCall(
            call_id=call_id, name=name, arguments=dict(arguments)
        )

    def add_tool_call_delta(
        self,
        index: int,
        *,
        call_id: str | None = None,
        name: str | None = None,
        arguments_delta: str = "",
    ) -> None:
        """Record a fragment of a tool call being streamed.

        Args:
            index: Which call this belongs to. Models emit **parallel** calls, and
                their fragments interleave, so the index is what keeps two calls'
                argument strings from being concatenated into one unparseable mess.
            call_id: The provider's id. Sent once, on the first fragment.
            name: The tool name. Sent once, on the first fragment.
            arguments_delta: A fragment of the JSON arguments. Rarely valid JSON on
                its own — that is why it is accumulated and parsed at the end.
        """
        partial = self._calls.setdefault(index, _PartialCall())
        if call_id:
            partial.call_id = call_id
        if name:
            partial.name = name
        if arguments_delta:
            partial.raw_arguments += arguments_delta

    # -- finishing ---------------------------------------------------------- #

    def build(
        self,
        *,
        model: str | None = None,
        usage: Usage | None = None,
        finish_reason: FinishReason = FinishReason.STOP,
        provider_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Produce the assembled response.

        Args:
            model: The model that actually served the request, if the provider
                said. Providers alias and silently upgrade, so what came back is
                worth recording over what was asked for.
            usage: Tokens consumed.
            finish_reason: Why generation stopped.
            provider_metadata: Opaque, adapter-owned.

        Returns:
            The assistant turn, its reasoning, its text and its tool calls.
        """
        parts: list[ContentPart] = []

        reasoning = "".join(self._reasoning)
        if reasoning or self._signature:
            # Reasoning first, mirroring the order every provider emits it in — and
            # the order Anthropic requires when it is echoed back on the next turn.
            parts.append(ReasoningPart(text=reasoning, signature=self._signature))

        text = "".join(self._text)
        if text:
            parts.append(TextPart(text=text))

        calls = tuple(self._finish(partial) for _, partial in sorted(self._calls.items()))

        return LLMResponse(
            message=Message(role=Role.ASSISTANT, content=tuple(parts), tool_calls=calls),
            model=model or self.model,
            usage=usage or Usage(),
            finish_reason=finish_reason,
            provider_metadata=dict(provider_metadata or {}),
        )

    @staticmethod
    def _finish(partial: _PartialCall) -> ToolCall:
        """Turn an accumulated call into a :class:`~sarb.contracts.ToolCall`."""
        if partial.arguments is not None:
            arguments, malformed = partial.arguments, None
        else:
            arguments, malformed = parse_arguments(partial.raw_arguments)

        return ToolCall(
            # Ollama and some OpenAI-compatible servers do not issue call ids at
            # all. One is minted rather than left empty, because the id is how a
            # result is matched back to its call — an empty one silently pairs
            # every parallel call with every result.
            id=partial.call_id or new_id(),
            name=partial.name or "",
            arguments=arguments,
            malformed_arguments=malformed,
        )
