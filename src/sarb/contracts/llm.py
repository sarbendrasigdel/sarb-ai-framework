"""The LLM port: request, response, streaming.

Every engine in the framework talks to models through :class:`LLMProvider`. No
engine knows that Anthropic, OpenAI, Gemini or Ollama exist.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal, Protocol, Self, runtime_checkable

from pydantic import ConfigDict, Field, model_validator

from sarb.contracts.base import SarbModel
from sarb.contracts.component import Family
from sarb.contracts.message import FinishReason, Message, ToolCall, Usage
from sarb.contracts.tool import ToolSpec

__all__ = [
    "LLMS",
    "LLMCapabilities",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "ReasoningDelta",
    "ResponseFormat",
    "StreamEnd",
    "StreamEvent",
    "StreamStart",
    "TextDelta",
    "TokenCounter",
    "ToolCallDelta",
    "ToolChoice",
]


class ToolChoice(SarbModel):
    """How the model should decide whether to call a tool.

    A model rather than a bare string, because ``"required"`` and "must call
    *this* tool" are different instructions and a plain string cannot say the
    second one without ambiguity — is ``"search"`` a mode or a tool name?

    Example:
        >>> ToolChoice.auto()          # the model decides (the default)
        >>> ToolChoice.required()      # it must call something
        >>> ToolChoice.none()          # it must not call anything
        >>> ToolChoice.tool("search")  # it must call `search`
    """

    mode: Literal["auto", "none", "required", "tool"] = "auto"
    name: str | None = None

    @classmethod
    def auto(cls) -> ToolChoice:
        """The model decides."""
        return cls(mode="auto")

    @classmethod
    def none(cls) -> ToolChoice:
        """The model must not call a tool."""
        return cls(mode="none")

    @classmethod
    def required(cls) -> ToolChoice:
        """The model must call some tool."""
        return cls(mode="required")

    @classmethod
    def tool(cls, name: str) -> ToolChoice:
        """The model must call this specific tool."""
        return cls(mode="tool", name=name)

    @model_validator(mode="after")
    def _name_only_with_tool_mode(self) -> Self:
        """A named tool needs ``mode="tool"``, and vice versa."""
        if (self.mode == "tool") != (self.name is not None):
            raise ValueError(
                'ToolChoice: name= is required for mode="tool", and forbidden otherwise'
            )
        return self


class ResponseFormat(SarbModel):
    """The shape the model's output must take.

    Attributes:
        kind: ``text`` (default), ``json`` (any valid JSON), or ``json_schema``.
        schema_: The JSON Schema, when ``kind`` is ``json_schema``. Named with a
            trailing underscore because ``schema`` collides with a Pydantic method;
            it serialises as ``schema``.
    """

    # Spelled out, not merged: the pydantic mypy plugin reads this statically and
    # cannot see through a dict-merge expression. See Step in sarb.contracts.workflow.
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        populate_by_name=True,
    )

    kind: Literal["text", "json", "json_schema"] = "text"
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")

    @model_validator(mode="after")
    def _schema_matches_kind(self) -> Self:
        """A ``json_schema`` format without a schema is a silent no-op otherwise."""
        if (self.kind == "json_schema") != (self.schema_ is not None):
            raise ValueError('ResponseFormat: schema= is required for kind="json_schema" only')
        return self


class LLMRequest(SarbModel):
    """One call to a model.

    Attributes:
        messages: The conversation so far, oldest first.
        system: The system prompt.
        tools: Tools the model may call.
        tool_choice: Whether and which it must call.
        max_tokens: Cap on output tokens.
        temperature: Sampling temperature. ``None`` means the provider's default,
            which is *not* the same as ``0.0`` — so the field is nullable rather
            than defaulted, and the framework never silently picks a value that
            changes a model's behaviour.
        stop: Sequences that end generation.
        seed: For reproducibility, where the provider supports it. Best-effort
            everywhere; no provider truly guarantees it.
        response_format: Structured-output constraint.
        metadata: Passed to the provider for its own tracking (request tagging,
            abuse monitoring). Not read by the framework.

    Why ``system`` is a field and not a message:
        Anthropic takes the system prompt as a top-level parameter; OpenAI takes
        it as a message. Modelling it as a field maps cleanly onto both, and — more
        importantly — makes "a system message halfway through a conversation"
        unrepresentable, because it is not a meaningful thing to ask for.
    """

    messages: tuple[Message, ...]
    system: str | None = None
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = Field(default_factory=ToolChoice.auto)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: tuple[str, ...] = ()
    seed: int | None = None
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_coherence(self) -> Self:
        """Reject requests that no provider could satisfy."""
        if not self.messages:
            raise ValueError("LLMRequest needs at least one message")

        if self.tool_choice.mode != "auto" and not self.tools:
            raise ValueError(f'tool_choice mode "{self.tool_choice.mode}" needs tools=')

        if self.tool_choice.mode == "tool":
            available = {tool.name for tool in self.tools}
            if self.tool_choice.name not in available:
                raise ValueError(
                    f"tool_choice names {self.tool_choice.name!r}, which is not in tools= "
                    f"({', '.join(sorted(available)) or 'none'})"
                )
        return self


class LLMResponse(SarbModel):
    """What a model returned.

    Attributes:
        message: The assistant turn, including any tool calls.
        model: The model that actually served the request, which is not always the
            one asked for — providers alias and silently upgrade. Worth recording.
        usage: Tokens consumed.
        finish_reason: Why it stopped. ``LENGTH`` means the answer is truncated.
        provider_metadata: Opaque, adapter-owned.
    """

    message: Message
    model: str
    usage: Usage = Field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.STOP
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """The response text. The common case, made cheap."""
        return self.message.text

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Tools the model wants to call."""
        return self.message.tool_calls


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


class StreamStart(SarbModel):
    """The provider has accepted the request and begun generating."""

    kind: Literal["start"] = "start"
    model: str


class TextDelta(SarbModel):
    """A fragment of the answer."""

    kind: Literal["text"] = "text"
    text: str


class ReasoningDelta(SarbModel):
    """A fragment of the model's exposed reasoning.

    A separate event type from :class:`TextDelta` on purpose: a UI that streams
    reasoning into the answer box is showing the user the model's scratch work as
    if it were the answer. Making them different types means you have to opt in.
    """

    kind: Literal["reasoning"] = "reasoning"
    text: str


class ToolCallDelta(SarbModel):
    """A fragment of a tool call being assembled.

    Providers stream tool arguments as partial JSON, so ``arguments_delta`` is a
    string fragment and not a dict — it is usually not valid JSON on its own.
    Callers that want whole tool calls should wait for :class:`StreamEnd`, which
    carries the assembled message.

    Attributes:
        index: Which call this belongs to, for parallel calls.
        id: The call id, sent once, on the first fragment.
        name: The tool name, sent once, on the first fragment.
        arguments_delta: A fragment of the JSON arguments.
    """

    kind: Literal["tool_call"] = "tool_call"
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


class StreamEnd(SarbModel):
    """Generation finished, carrying the fully assembled response.

    This is what makes streaming tolerable to consume. Without it, every caller
    that needs the whole message — to append to history, to dispatch tool calls, to
    count tokens — has to reassemble it from deltas, and every caller would do it
    slightly differently. The adapter assembles it once, from its provider's own
    terminal event, which it is far better placed to do correctly.
    """

    kind: Literal["end"] = "end"
    response: LLMResponse


#: One event in a stream. Discriminated on ``kind``, so a consumer can match
#: exhaustively and a new event type does not silently fall through.
type StreamEvent = Annotated[
    StreamStart | TextDelta | ReasoningDelta | ToolCallDelta | StreamEnd,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #


class LLMCapabilities(SarbModel):
    """What a configured model can actually do.

    Lets an engine fail fast and clearly — "this model cannot call tools" at wiring
    time — instead of sending a request that a provider rejects with a 400 whose
    message assumes you know its API.

    Attributes:
        streaming: Supports token streaming.
        tools: Supports tool calling.
        vision: Accepts image content.
        json_schema: Supports schema-constrained output.
        reasoning: Exposes reasoning content.
        max_input_tokens: Context window, if known.
        max_output_tokens: Output cap, if known.
    """

    streaming: bool = True
    tools: bool = False
    vision: bool = False
    json_schema: bool = False
    reasoning: bool = False
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None


# --------------------------------------------------------------------------- #
# Ports
# --------------------------------------------------------------------------- #


@runtime_checkable
class LLMProvider(Protocol):
    """A language model.

    Implementations translate to and from one vendor's API and are the only place
    in the framework that names one.

    Adapters raise from the :mod:`sarb.errors` hierarchy — :class:`~sarb.errors.LLMError`,
    :class:`~sarb.errors.RateLimitError`, :class:`~sarb.errors.AuthenticationError` —
    never a vendor's exception class. An engine handling a rate limit must not have
    to import an SDK to catch it.
    """

    @property
    def model(self) -> str:
        """The model this provider is configured to call."""
        ...

    @property
    def capabilities(self) -> LLMCapabilities:
        """What this model supports."""
        ...

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a complete response.

        Args:
            request: The conversation and parameters.

        Returns:
            The assistant's turn, with usage and finish reason.

        Raises:
            LLMError: The call failed.
            RateLimitError: The provider is rate limiting. Retryable.
            AuthenticationError: Credentials were rejected. Not retryable.
        """
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Stream a response.

        Not an ``async def``: it returns an async iterator rather than awaiting one,
        so callers write ``async for event in llm.stream(request)`` without an extra
        ``await``.

        The final event is always :class:`StreamEnd`, carrying the assembled
        response — even when the stream produced no text.

        Args:
            request: The conversation and parameters.

        Yields:
            Stream events, ending with :class:`StreamEnd`.

        Raises:
            LLMError: The call failed. A failure part-way through a stream raises
                from the iterator; content already yielded stands.
        """
        ...


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens without calling the model.

    Deliberately a *separate* port. Only some providers can do this locally, and
    forcing it onto :class:`LLMProvider` would make every adapter implement a method
    most of them can only fake. Interface segregation: an engine that needs counting
    checks for it.

    Example:
        >>> if isinstance(llm, TokenCounter):
        ...     if await llm.count_tokens(request) > budget:
        ...         messages = trim(messages)
    """

    async def count_tokens(self, request: LLMRequest) -> int:
        """Count the input tokens this request would consume."""
        ...


#: The ``llms`` config family.
LLMS = Family("llms", LLMProvider)
