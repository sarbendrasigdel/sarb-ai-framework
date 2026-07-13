"""Claude, through Anthropic's Messages API.

The interesting part of this file is not the HTTP call — the SDK does that. It is
the translation, in both directions, between the framework's neutral
:class:`~sarb.contracts.Message` and Anthropic's content blocks. Three places
where the two shapes genuinely disagree, and what is done about each:

**Tool results.** The framework models one tool turn carrying *many* results.
Anthropic agrees — ``tool_result`` blocks go in a single ``user`` message — so this
adapter has the easy side of that argument. The OpenAI adapter, which must fan the
same turn out into one message per result, has the hard side. That asymmetry is
exactly what an adapter is for.

**Reasoning.** Anthropic returns ``thinking`` blocks with a ``signature``, and
requires them echoed back **unchanged** on the next turn of the same conversation.
The framework carries them as :class:`~sarb.contracts.ReasoningPart`; this adapter
is careful to round-trip the signature verbatim, and to drop unsigned reasoning
rather than send something the API will reject.

**The system prompt.** A top-level parameter here, a message elsewhere. The
framework chose the parameter, so this is a straight assignment. See
``docs/architecture/ports.md`` for why.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, SecretStr

from sarb.config import parse_options
from sarb.contracts import (
    FinishReason,
    ImagePart,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    Message,
    ReasoningDelta,
    ReasoningPart,
    Role,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextPart,
    ToolCallDelta,
    ToolChoice,
    ToolSpec,
    Usage,
)
from sarb.dependencies import import_optional
from sarb.errors import LLMError, ProviderTimeoutError, SarbError
from sarb.providers._shared import error_for_status, retry_after_seconds, unsupported
from sarb.providers.llm._shared import LLMOptions, ResponseBuilder

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from anthropic import AsyncAnthropic

    from sarb.plugins.registry import ComponentContext

__all__ = ["AnthropicOptions", "AnthropicProvider", "build"]

PROVIDER = "anthropic"
EXTRA = "anthropic"


class AnthropicOptions(LLMOptions):
    """Configuration for the ``anthropic`` LLM provider.

    Attributes:
        api_key: The API key. Omit it and the SDK reads ``ANTHROPIC_API_KEY``,
            which is what a container deployment should do — a secret in YAML is a
            secret in a git history.
        max_tokens: Output cap used when a request does not set one. Anthropic
            **requires** ``max_tokens`` on every call, so unlike other providers
            this adapter cannot simply omit it, and a default has to exist.
        thinking: Turn on adaptive thinking. The model decides how much to reason;
            there is no token budget to tune. Reasoning arrives as
            :class:`~sarb.contracts.ReasoningPart` and
            :class:`~sarb.contracts.ReasoningDelta`.
        reasoning_display: Whether reasoning text is returned at all. Anthropic's
            default is ``omitted`` — the blocks arrive with empty text — which to a
            streaming UI looks like a long pause before anything happens. This
            adapter defaults to ``summarized`` instead, because a framework whose
            ``reasoning=True`` capability yields no reasoning is lying.
        effort: How hard the model works. Trades quality against latency and spend.

    Example:
        ```yaml
        llms:
          default:
            provider: anthropic
            options:
              model: claude-opus-4-8
              max_tokens: 8192
              thinking: true
              effort: high
        ```
    """

    api_key: SecretStr | None = None
    max_tokens: int = Field(default=4096, gt=0)
    thinking: bool = False
    reasoning_display: Literal["omitted", "summarized"] = "summarized"
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


class AnthropicProvider:
    """An :class:`~sarb.contracts.LLMProvider` backed by Anthropic.

    The client is injected rather than built here. That is what makes the
    translation — which is where every real bug in an adapter lives — testable
    against Anthropic's own response models, with no key and no network. Use
    :func:`build` to construct one from configuration.
    """

    #: Anthropic's stop reasons, mapped onto ours. ``pause_turn`` means a
    #: server-side tool is still working and the turn can be resumed; from the
    #: framework's point of view the model simply stopped talking.
    _FINISH: ClassVar[dict[str, FinishReason]] = {
        "end_turn": FinishReason.STOP,
        "stop_sequence": FinishReason.STOP,
        "pause_turn": FinishReason.STOP,
        "max_tokens": FinishReason.LENGTH,
        "tool_use": FinishReason.TOOL_CALLS,
        "refusal": FinishReason.CONTENT_FILTER,
    }

    _TOOL_CHOICE: ClassVar[dict[str, str]] = {
        "auto": "auto",
        "none": "none",
        "required": "any",  # Anthropic calls it "any"; the meaning is identical.
    }

    def __init__(self, client: AsyncAnthropic, options: AnthropicOptions) -> None:
        """Wrap an Anthropic client.

        Args:
            client: An ``anthropic.AsyncAnthropic``.
            options: Validated configuration.
        """
        self._client = client
        self._options = options
        # Held so the error path does not import anything. Constructing this class
        # at all implies the SDK is present, so this cannot fail here.
        self._sdk = import_optional("anthropic", extra=EXTRA, used_by=f"LLM provider {PROVIDER!r}")

    # -- port --------------------------------------------------------------- #

    @property
    def model(self) -> str:
        """The configured model."""
        return self._options.model

    @property
    def capabilities(self) -> LLMCapabilities:
        """What the Messages API supports.

        Note what is *not* claimed: this describes the API, not the model. Whether
        the configured model has a 200K or a 1M context window is
        :attr:`~sarb.providers.llm._shared.LLMOptions.max_input_tokens`, because
        only an operator knows, and a table shipped in the framework would be stale
        within a quarter.
        """
        return LLMCapabilities(
            streaming=True,
            tools=True,
            vision=True,
            json_schema=True,
            reasoning=self._options.thinking,
            max_input_tokens=self._options.max_input_tokens,
            max_output_tokens=self._options.max_output_tokens,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Call the model and return the whole response."""
        params = self._params(request)
        try:
            message = await self._client.messages.create(**params)
        except Exception as exc:
            raise self._fail(exc) from exc

        return self._to_response(message)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Call the model and yield events as they arrive.

        Raw Anthropic events are translated here rather than leaning on the SDK's
        own stream-accumulator helper. The helper is convenient and it is also a
        second, undocumented contract that changes between SDK versions; the raw
        event names (``content_block_delta``, ``input_json_delta``) are the wire
        format, and they do not.
        """
        params = self._params(request)
        builder = ResponseBuilder(self._options.model)
        model = self._options.model
        usage = Usage()
        finish = FinishReason.STOP
        started = False

        try:
            events = await self._client.messages.create(**params, stream=True)
            async for event in events:
                if event.type == "message_start":
                    model = event.message.model or model
                    usage = self._usage(event.message.usage)
                    started = True
                    yield StreamStart(model=model)

                elif event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        builder.add_tool_call_delta(event.index, call_id=block.id, name=block.name)
                        yield ToolCallDelta(index=event.index, id=block.id, name=block.name)

                elif event.type == "content_block_delta":
                    for translated in self._delta(event, builder):
                        yield translated

                elif event.type == "message_delta":
                    if event.delta.stop_reason:
                        finish = self._FINISH.get(event.delta.stop_reason, FinishReason.STOP)
                    # Anthropic reports the *total* output tokens here, not an
                    # increment — so this replaces rather than adds. Summing would
                    # double-count the handful already reported at message_start.
                    usage = usage.evolve(
                        output_tokens=getattr(event.usage, "output_tokens", 0) or 0
                    )
        except Exception as exc:
            raise self._fail(exc) from exc

        if not started:
            yield StreamStart(model=model)

        yield StreamEnd(response=builder.build(model=model, usage=usage, finish_reason=finish))

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Called by the container on shutdown."""
        await self._client.close()

    # -- outbound translation ----------------------------------------------- #

    def _params(self, request: LLMRequest) -> dict[str, Any]:
        """Turn an :class:`~sarb.contracts.LLMRequest` into Messages API kwargs."""
        options = self._options
        params: dict[str, Any] = {
            "model": options.model,
            "max_tokens": request.max_tokens or options.max_tokens,
            "messages": [self._message(message) for message in request.messages],
        }

        if request.system:
            params["system"] = request.system

        if request.tools:
            params["tools"] = [self._tool(tool) for tool in request.tools]
            params["tool_choice"] = self._tool_choice(request.tool_choice)

        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop:
            params["stop_sequences"] = list(request.stop)

        # `seed` is absent from Anthropic's API. The port documents seeding as
        # best-effort and explicitly not guaranteed anywhere, so dropping it is
        # within contract — unlike response_format below, which is not.

        if options.thinking:
            params["thinking"] = {"type": "adaptive", "display": options.reasoning_display}

        output_config: dict[str, Any] = {}
        if options.effort:
            output_config["effort"] = options.effort

        fmt = request.response_format
        if fmt.kind == "json_schema":
            output_config["format"] = {"type": "json_schema", "schema": fmt.schema_}
        elif fmt.kind == "json":
            raise unsupported(
                PROVIDER,
                'response_format kind="json"',
                hint='Anthropic constrains output with a schema. Use kind="json_schema".',
            )

        if output_config:
            params["output_config"] = output_config

        # Anthropic's `metadata` accepts one key, and inventing others is a 400.
        user_id = request.metadata.get("user_id")
        if user_id:
            params["metadata"] = {"user_id": user_id}

        return params

    def _message(self, message: Message) -> dict[str, Any]:
        """Render one framework message as an Anthropic message."""
        if message.role is Role.TOOL:
            # Anthropic has no `tool` role: results come back as blocks in a user
            # turn. The framework's one-turn-many-results shape maps straight onto
            # it, which is the case OpenAI's adapter has to work for.
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                    for result in message.tool_results
                ],
            }

        content: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, ReasoningPart):
                # Unsigned reasoning is dropped rather than sent. Anthropic
                # validates the signature and rejects the turn without it, so
                # forwarding reasoning that came from *another* provider — which is
                # exactly what a framework makes easy — would break the request.
                if part.signature:
                    content.append(
                        {
                            "type": "thinking",
                            "thinking": part.text,
                            "signature": part.signature,
                        }
                    )
            elif isinstance(part, TextPart):
                if part.text:
                    content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                content.append({"type": "image", "source": self._image_source(part)})

        for call in message.tool_calls:
            content.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )

        return {"role": message.role.value, "content": content}

    @staticmethod
    def _image_source(part: ImagePart) -> dict[str, Any]:
        """Render an image as an Anthropic image source."""
        if part.url is not None:
            return {"type": "url", "url": part.url}

        data = base64.standard_b64encode(part.data or b"").decode("ascii")
        return {"type": "base64", "media_type": part.media_type, "data": data}

    @staticmethod
    def _tool(spec: ToolSpec) -> dict[str, Any]:
        """Render a tool spec as an Anthropic tool definition."""
        tool: dict[str, Any] = {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.parameters,
        }
        if spec.strict:
            tool["strict"] = True
        return tool

    @classmethod
    def _tool_choice(cls, choice: ToolChoice) -> dict[str, Any]:
        """Render a tool choice."""
        if choice.mode == "tool":
            return {"type": "tool", "name": choice.name}
        return {"type": cls._TOOL_CHOICE[choice.mode]}

    # -- inbound translation ------------------------------------------------ #

    def _to_response(self, message: Any) -> LLMResponse:
        """Turn an Anthropic message into an :class:`~sarb.contracts.LLMResponse`."""
        builder = ResponseBuilder(self._options.model)

        for block in message.content:
            if block.type == "text":
                builder.add_text(block.text)
            elif block.type == "thinking":
                builder.add_reasoning(block.thinking or "", signature=block.signature)
            elif block.type == "tool_use":
                builder.add_tool_call(call_id=block.id, name=block.name, arguments=block.input)

        return builder.build(
            model=message.model or self._options.model,
            usage=self._usage(message.usage),
            finish_reason=self._FINISH.get(message.stop_reason or "", FinishReason.STOP),
            provider_metadata={"id": message.id},
        )

    def _delta(self, event: Any, builder: ResponseBuilder) -> list[StreamEvent]:
        """Translate one ``content_block_delta``, updating the builder as it goes."""
        delta = event.delta

        if delta.type == "text_delta":
            builder.add_text(delta.text)
            return [TextDelta(text=delta.text)]

        if delta.type == "thinking_delta":
            builder.add_reasoning(delta.thinking)
            return [ReasoningDelta(text=delta.thinking)]

        if delta.type == "signature_delta":
            # No event of our own: a signature is a token to echo back, not content
            # anyone would render. It still has to reach the assembled response, or
            # the next turn of the conversation is rejected.
            builder.set_signature(delta.signature)
            return []

        if delta.type == "input_json_delta":
            builder.add_tool_call_delta(event.index, arguments_delta=delta.partial_json)
            return [ToolCallDelta(index=event.index, arguments_delta=delta.partial_json)]

        return []

    @staticmethod
    def _usage(usage: Any) -> Usage:
        """Read Anthropic's usage, whichever of its shapes this one is.

        Streaming reports usage twice, in two different types — a full one at
        ``message_start`` and a partial one at ``message_delta`` — so every field is
        read defensively rather than assumed present.

        **Anthropic's ``input_tokens`` is not our ``input_tokens``.** Theirs counts only
        the tokens after the last cache breakpoint: the documented total is
        ``input_tokens + cache_read_input_tokens + cache_creation_input_tokens``. Ours is
        the total, because a field whose meaning depends on the vendor is a field nobody
        can bill against. So the three are added up here, once, rather than every caller
        downstream discovering the convention for itself.
        """
        if usage is None:
            return Usage()

        details = getattr(usage, "output_tokens_details", None)
        fresh = getattr(usage, "input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        return Usage(
            input_tokens=fresh + cache_read + cache_write,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cached_input_tokens=cache_read,
            cache_write_input_tokens=cache_write,
            # Thinking tokens are already inside Anthropic's output_tokens, which is
            # what our contract asks for, so this is a straight read.
            reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
        )

    # -- errors ------------------------------------------------------------- #

    def _fail(self, exc: BaseException) -> BaseException:
        """Translate an Anthropic exception into the framework's hierarchy.

        An engine handling a rate limit must never have to ``import anthropic`` to
        catch it. This is the only place in the framework that knows what
        ``anthropic.RateLimitError`` is.
        """
        if isinstance(exc, SarbError):
            return exc

        sdk = self._sdk
        model = self._options.model

        if isinstance(exc, sdk.APITimeoutError):
            return ProviderTimeoutError(
                f"Anthropic did not respond within {self._options.timeout}s",
                provider=PROVIDER,
                context={"model": model, "timeout": self._options.timeout},
                cause=exc,
            )

        if isinstance(exc, sdk.APIConnectionError):
            return LLMError(
                "Could not reach Anthropic",
                provider=PROVIDER,
                context={"model": model},
                cause=exc,
                retryable=True,
            )

        if isinstance(exc, sdk.APIStatusError):
            return error_for_status(
                exc.status_code,
                str(exc),
                provider=PROVIDER,
                model=model,
                otherwise=LLMError,
                retry_after=retry_after_seconds(getattr(exc.response, "headers", None)),
                cause=exc,
            )

        return LLMError(
            f"Anthropic call failed: {exc}",
            provider=PROVIDER,
            context={"model": model},
            cause=exc,
        )


def build(ctx: ComponentContext) -> AnthropicProvider:
    """Build the ``anthropic`` LLM provider from configuration.

    The SDK is imported here and nowhere else, so importing this module on a
    core-only install is free.
    """
    sdk = import_optional("anthropic", extra=EXTRA, used_by=f"LLM provider {PROVIDER!r}")
    options = parse_options(ctx.config, AnthropicOptions, component=ctx.path)

    client = sdk.AsyncAnthropic(
        api_key=options.api_key.get_secret_value() if options.api_key else None,
        base_url=options.base_url,
        timeout=options.timeout,
        max_retries=options.max_retries,
    )
    return AnthropicProvider(client, options)
