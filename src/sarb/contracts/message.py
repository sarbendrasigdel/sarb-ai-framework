"""Conversation: messages, content parts, tool calls, token usage.

This is the most consequential model in the framework — every engine reads it and
every LLM adapter must translate it — so the shape is argued for rather than
assumed.

Why content is a list of parts
------------------------------
``content: str`` cannot express an image, and bolting on a parallel ``images``
field means every consumer must handle two representations of "what the user
said". A sequence of typed parts handles text, images and model reasoning
uniformly, and leaves room for part kinds nobody has invented yet.

The cost is that the common case — a line of text — becomes verbose. That is paid
back with constructors: ``Message.user("hello")``, and ``message.text`` to read it
back out.

Why tool calls are fields, not content parts
--------------------------------------------
Anthropic models tool use as blocks *inside* content; OpenAI models it as a
separate field on the message. Either can represent the other, so the question is
which reads better in the code that actually consumes it — and that code is the
agent loop, where ``if message.tool_calls:`` is plainly better than scanning a
content sequence for parts of a particular kind. The adapters absorb the
difference. That is their job.

Why a tool message holds *many* results
---------------------------------------
Models emit parallel tool calls. OpenAI wants one message per result; Anthropic
wants one message containing every result. Modelling one message with a tuple of
results keeps the framework's own history compact and honest — one model turn,
one response turn — and leaves the fan-out to the OpenAI adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from sarb.contracts.base import SarbModel

__all__ = [
    "ContentPart",
    "FinishReason",
    "ImagePart",
    "Message",
    "ReasoningPart",
    "Role",
    "TextPart",
    "ToolCall",
    "ToolResult",
    "Usage",
]


class Role(StrEnum):
    """Who produced a message.

    There is deliberately **no** ``SYSTEM`` role. The system prompt is a parameter
    of a request, not a turn in a conversation — see
    :attr:`~sarb.contracts.llm.LLMRequest.system`. Allowing both would give the
    framework two ways to say the same thing, and invite the undefined question of
    what a system message *in the middle* of a conversation means.
    """

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #


class TextPart(SarbModel):
    """A run of text."""

    kind: Literal["text"] = "text"
    text: str


class ImagePart(SarbModel):
    """An image, inline or by reference.

    Attributes:
        data: Raw bytes, for an image the application already holds.
        url: A URL the provider fetches. Exactly one of ``data`` or ``url``.
        media_type: IANA type, e.g. ``image/png``. Providers require it and do
            not reliably sniff it.
    """

    kind: Literal["image"] = "image"
    data: bytes | None = None
    url: str | None = None
    media_type: str = "image/png"

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Self:
        """An image with both or neither source is a bug, caught here."""
        if (self.data is None) == (self.url is None):
            raise ValueError("ImagePart needs exactly one of data= or url=")
        return self


class ReasoningPart(SarbModel):
    """A model's exposed reasoning.

    Attributes:
        text: The reasoning, where the provider exposes it.
        signature: An opaque token some providers require to be echoed back on the
            next turn for the reasoning to remain valid. The framework never reads
            it; it only carries it, which is precisely why it is typed as opaque.
    """

    kind: Literal["reasoning"] = "reasoning"
    text: str = ""
    signature: str | None = None


#: A piece of a message. Discriminated on ``kind``, so adding a part type later
#: is additive and old data still parses.
type ContentPart = Annotated[TextPart | ImagePart | ReasoningPart, Field(discriminator="kind")]


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


class ToolCall(SarbModel):
    """A model's request to invoke a tool.

    Attributes:
        id: The provider's id for this call. Echoed back on the result so parallel
            calls can be matched to their results.
        name: The tool to invoke.
        arguments: The parsed arguments.
        malformed_arguments: The raw string, when the model emitted something that
            was not valid JSON.

    The ``malformed_arguments`` field is the important one, and exists because
    models do occasionally emit broken JSON. The obvious designs both fail: raising
    inside the adapter turns a *recoverable* event into a dead request, and
    silently coercing to ``{}`` invents an intent the model never had. Carrying the
    raw text lets the tool engine hand the model back an error it can retry from —
    which is what a human would do, and what the model is perfectly capable of
    doing.
    """

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    malformed_arguments: str | None = None

    @property
    def is_malformed(self) -> bool:
        """Whether the model's arguments failed to parse."""
        return self.malformed_arguments is not None


class ToolResult(SarbModel):
    """The outcome of one tool invocation, on its way back to the model.

    Attributes:
        call_id: The :attr:`ToolCall.id` this answers.
        name: The tool that ran.
        content: What the model should see. Rendered text, not a Python object —
            the model reads it.
        is_error: Whether the tool failed. A failed tool is normally *not* a failed
            request: the error text goes back to the model, which can correct
            itself. See :class:`~sarb.errors.ToolError`.

    Known limitation: ``content`` is text. Tools that return images (screenshots,
    charts) will need a parts sequence here, exactly as :class:`Message` has. That
    change is additive and deliberately deferred until there is a caller for it.
    """

    call_id: str
    name: str
    content: str
    is_error: bool = False


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #


class Usage(SarbModel):
    """Tokens consumed by one call.

    **The totals are inclusive, and every adapter must make them so.** This is the
    single most important sentence in this file, because the vendors do not agree and
    the disagreement is invisible:

    * OpenAI's ``prompt_tokens`` **includes** its cached tokens.
    * Anthropic's ``input_tokens`` **excludes** them —
      ``total = input + cache_read + cache_creation``.
    * Gemini bills reasoning as *"the sum of output tokens and thinking tokens"*, so its
      thinking tokens sit **outside** its ``candidates_token_count``, while OpenAI's and
      Anthropic's sit **inside** theirs.

    Copy each vendor's number straight into the same field and this model means three
    different things depending on who served the call — and a cost figure derived from
    it is silently wrong for two of the three, in different directions. So the shape is
    pinned here and the adapter is what does the arithmetic:

    ``input_tokens``
        **Every** input token, including any read from or written to a cache.
    ``cached_input_tokens``, ``cache_write_input_tokens``
        Subsets of ``input_tokens``, split out because they are billed differently —
        a read is far cheaper than fresh input, a write is dearer.
    ``output_tokens``
        **Every** generated token, including reasoning.
    ``reasoning_tokens``
        A subset of ``output_tokens``.

    The invariants are validated rather than trusted, because an adapter that gets this
    wrong produces numbers that look entirely plausible.

    Deliberately tokens only, not cost. Cost is a function of tokens *and* a price list
    that changes without warning and differs per contract; baking it in here would put a
    number in the domain model that is wrong the day a vendor updates its pricing page.
    See :mod:`sarb.middleware.cost`, where the price list is configuration.
    """

    input_tokens: int = Field(
        default=0, ge=0, description="All input tokens, cache reads and writes included."
    )
    output_tokens: int = Field(
        default=0, ge=0, description="All generated tokens, reasoning included."
    )
    cached_input_tokens: int = Field(
        default=0,
        ge=0,
        description="Of `input_tokens`, those served from a provider-side cache.",
    )
    cache_write_input_tokens: int = Field(
        default=0,
        ge=0,
        description="Of `input_tokens`, those written *into* a provider-side cache.",
    )
    reasoning_tokens: int = Field(
        default=0, ge=0, description="Of `output_tokens`, those spent reasoning."
    )

    @model_validator(mode="after")
    def _check_the_parts_fit_inside_the_totals(self) -> Usage:
        """Catch an adapter that reported a vendor's numbers without normalising them.

        Without this, Anthropic's convention silently makes ``cached_input_tokens``
        larger than ``input_tokens`` on a cache hit, and nothing downstream notices —
        it just quietly bills a negative number of fresh input tokens.
        """
        cache = self.cached_input_tokens + self.cache_write_input_tokens
        if cache > self.input_tokens:
            raise ValueError(
                f"cached ({self.cached_input_tokens}) + cache writes "
                f"({self.cache_write_input_tokens}) exceed input_tokens ({self.input_tokens}). "
                "input_tokens must include them both."
            )
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError(
                f"reasoning_tokens ({self.reasoning_tokens}) exceed output_tokens "
                f"({self.output_tokens}). output_tokens must include them."
            )
        return self

    @property
    def uncached_input_tokens(self) -> int:
        """Input that was neither read from nor written to a cache — the full-price part."""
        return self.input_tokens - self.cached_input_tokens - self.cache_write_input_tokens

    @property
    def total_tokens(self) -> int:
        """Input plus output. Cached and reasoning tokens are already inside those."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        """Sum usage across calls.

        An agent run makes many model calls, and the number anyone actually wants
        is the total for the run — so summing is a first-class operation rather
        than something every caller reimplements.
        """
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class FinishReason(StrEnum):
    """Why the model stopped generating."""

    STOP = "stop"
    """The model finished its turn."""

    LENGTH = "length"
    """It hit the token limit. The answer is truncated — usually a bug, not an answer."""

    TOOL_CALLS = "tool_calls"
    """It wants to call tools and is waiting for results."""

    CONTENT_FILTER = "content_filter"
    """The provider blocked the output."""

    ERROR = "error"
    """Generation failed part-way. Any partial content is not trustworthy."""


# --------------------------------------------------------------------------- #
# Message
# --------------------------------------------------------------------------- #


class Message(SarbModel):
    """One turn in a conversation.

    Attributes:
        role: Who produced it.
        content: Its parts, in order.
        tool_calls: Tools the model wants to invoke. Assistant messages only.
        tool_results: Results being returned to the model. Tool messages only, and
            plural because models emit parallel calls.
        name: An optional speaker name, for multi-agent transcripts.
        provider_metadata: Opaque, adapter-owned data that must survive a round
            trip — a reasoning signature, a cache marker. The framework never reads
            it and never reasons about it; it only carries it. Only the adapter
            that wrote a key may read that key.

    Example:
        >>> Message.user("What is the capital of France?")
        >>> Message.assistant("Paris.")
        >>> Message.tool([ToolResult(call_id="c1", name="search", content="...")])
    """

    role: Role
    content: tuple[ContentPart, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    name: str | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    # -- constructors ------------------------------------------------------- #

    @classmethod
    def user(cls, text: str, **kwargs: Any) -> Message:
        """A user turn of plain text."""
        return cls(role=Role.USER, content=(TextPart(text=text),), **kwargs)

    @classmethod
    def assistant(cls, text: str = "", **kwargs: Any) -> Message:
        """An assistant turn, optionally with text."""
        content = (TextPart(text=text),) if text else ()
        return cls(role=Role.ASSISTANT, content=content, **kwargs)

    @classmethod
    def tool(cls, results: Sequence[ToolResult]) -> Message:
        """A turn returning tool results to the model."""
        return cls(role=Role.TOOL, tool_results=tuple(results))

    # -- reading ------------------------------------------------------------ #

    @property
    def text(self) -> str:
        """Every text part, concatenated.

        The common case, made cheap. Reasoning parts are excluded: they are the
        model thinking, not the model answering, and showing them to a user as the
        answer would be wrong.
        """
        return "".join(part.text for part in self.content if isinstance(part, TextPart))

    @property
    def reasoning(self) -> str:
        """Every reasoning part, concatenated."""
        return "".join(part.text for part in self.content if isinstance(part, ReasoningPart))

    @property
    def images(self) -> tuple[ImagePart, ...]:
        """Every image part, in order."""
        return tuple(part for part in self.content if isinstance(part, ImagePart))

    # -- invariants --------------------------------------------------------- #

    @model_validator(mode="after")
    def _check_role_invariants(self) -> Self:
        """Reject messages that no provider could render.

        Enforced here, once, rather than in four adapters that would each discover
        the problem differently — and one of which would not check at all.
        """
        if self.tool_calls and self.role is not Role.ASSISTANT:
            raise ValueError("only an assistant message may carry tool_calls")

        if self.tool_results and self.role is not Role.TOOL:
            raise ValueError("only a tool message may carry tool_results")

        if self.role is Role.TOOL and not self.tool_results:
            raise ValueError("a tool message must carry at least one result")

        if self.role is Role.USER and not self.content:
            raise ValueError("a user message must have content")

        return self
