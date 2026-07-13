"""OpenAI, through the Chat Completions API.

This adapter also serves every **OpenAI-compatible** endpoint — vLLM, Ollama's
compatibility layer, Together, Groq, LiteLLM, an in-house gateway — by pointing
``base_url`` at them. That is a large part of its value: one adapter, most of the
inference market. It is also why two options here exist purely to absorb the ways
those servers differ from the real thing (see ``token_limit_field``).

Where OpenAI and the framework disagree
---------------------------------------
**Tool results fan out.** The framework models one tool turn carrying every result
from a round of parallel calls. OpenAI wants one ``tool`` message per result. So
``_messages`` is the only translator in the framework that is not one-to-one, and
that is deliberate: the neutral model stays honest — one model turn, one response
turn — and the fan-out happens exactly here, in the adapter that needs it.

**The system prompt is a message.** Prepended, not passed. Trivial in this
direction; it is the other direction that would have been lossy, which is why the
port made ``system`` a field.

**Reasoning is not returned.** Chat Completions bills reasoning tokens and does not
show them. The adapter reports the token count in
:attr:`~sarb.contracts.Usage.reasoning_tokens` and claims ``reasoning=False`` in
its capabilities, because a capability that means "you will be charged for
something you cannot see" is not a capability.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import SecretStr

from sarb.config import parse_options
from sarb.contracts import (
    FinishReason,
    ImagePart,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    Message,
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
from sarb.providers._shared import error_for_status, retry_after_seconds
from sarb.providers.llm._shared import LLMOptions, ResponseBuilder

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from openai import AsyncOpenAI

    from sarb.plugins.registry import ComponentContext

__all__ = ["OpenAIOptions", "OpenAIProvider", "build"]

PROVIDER = "openai"
EXTRA = "openai"


class OpenAIOptions(LLMOptions):
    """Configuration for the ``openai`` LLM provider.

    Attributes:
        api_key: The API key. Omit it and the SDK reads ``OPENAI_API_KEY``.
        organization: Optional organisation id for billing attribution.
        vision: Whether the configured model accepts images. Declared rather than
            guessed: OpenAI-compatible servers host arbitrary models, and there is
            no reliable way to ask one whether it has eyes.
        token_limit_field: Which parameter carries the output cap.
            ``max_completion_tokens`` is what OpenAI now requires and what reasoning
            models *only* accept; ``max_tokens`` is what most compatible servers
            still implement. Wrong choice, and the cap is silently dropped or the
            call is rejected — so it is a knob, not a guess.

    Example:
        ```yaml
        llms:
          default:
            provider: openai
            options:
              model: gpt-4o
              vision: true

          local:                          # a vLLM server, same adapter
            provider: openai
            options:
              model: meta-llama/Llama-3.1-8B-Instruct
              base_url: http://localhost:8000/v1
              api_key: not-used-but-required
              token_limit_field: max_tokens
        ```
    """

    api_key: SecretStr | None = None
    organization: str | None = None
    vision: bool = False
    token_limit_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"


class OpenAIProvider:
    """An :class:`~sarb.contracts.LLMProvider` backed by Chat Completions.

    Use :func:`build` to construct one from configuration.
    """

    _FINISH: ClassVar[dict[str, FinishReason]] = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "function_call": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
    }

    def __init__(self, client: AsyncOpenAI, options: OpenAIOptions) -> None:
        """Wrap an OpenAI client.

        Args:
            client: An ``openai.AsyncOpenAI``.
            options: Validated configuration.
        """
        self._client = client
        self._options = options
        self._sdk = import_optional("openai", extra=EXTRA, used_by=f"LLM provider {PROVIDER!r}")

    # -- port --------------------------------------------------------------- #

    @property
    def model(self) -> str:
        """The configured model."""
        return self._options.model

    @property
    def capabilities(self) -> LLMCapabilities:
        """What this endpoint supports."""
        return LLMCapabilities(
            streaming=True,
            tools=True,
            vision=self._options.vision,
            json_schema=True,
            # Chat Completions charges for reasoning tokens but never returns the
            # reasoning. Claiming the capability would promise content that never
            # arrives.
            reasoning=False,
            max_input_tokens=self._options.max_input_tokens,
            max_output_tokens=self._options.max_output_tokens,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Call the model and return the whole response."""
        params = self._params(request)
        try:
            completion = await self._client.chat.completions.create(**params)
        except Exception as exc:
            raise self._fail(exc) from exc

        return self._to_response(completion)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Call the model and yield events as they arrive."""
        params = self._params(request)
        builder = ResponseBuilder(self._options.model)
        model = self._options.model
        usage = Usage()
        finish = FinishReason.STOP
        started = False

        try:
            chunks = await self._client.chat.completions.create(
                **params,
                stream=True,
                # Without this, a streamed call reports no usage at all — and an
                # agent that cannot count its own tokens cannot budget.
                stream_options={"include_usage": True},
            )
            async for chunk in chunks:
                if not started:
                    model = chunk.model or model
                    started = True
                    yield StreamStart(model=model)

                if chunk.usage is not None:
                    usage = self._usage(chunk.usage)

                if not chunk.choices:
                    continue  # The usage-only chunk that arrives last.

                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish = self._FINISH.get(choice.finish_reason, FinishReason.STOP)

                delta = choice.delta
                if delta is None:
                    continue

                if delta.content:
                    builder.add_text(delta.content)
                    yield TextDelta(text=delta.content)

                for call in delta.tool_calls or ():
                    function = call.function
                    arguments = (function.arguments if function else None) or ""
                    name = function.name if function else None
                    builder.add_tool_call_delta(
                        call.index, call_id=call.id, name=name, arguments_delta=arguments
                    )
                    yield ToolCallDelta(
                        index=call.index,
                        id=call.id,
                        name=name,
                        arguments_delta=arguments,
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
        """Turn an :class:`~sarb.contracts.LLMRequest` into Chat Completions kwargs."""
        params: dict[str, Any] = {
            "model": self._options.model,
            "messages": self._messages(request),
        }

        if request.tools:
            params["tools"] = [self._tool(tool) for tool in request.tools]
            params["tool_choice"] = self._tool_choice(request.tool_choice)

        if request.max_tokens is not None:
            params[self._options.token_limit_field] = request.max_tokens
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop:
            params["stop"] = list(request.stop)
        if request.seed is not None:
            params["seed"] = request.seed

        fmt = request.response_format
        if fmt.kind == "json":
            params["response_format"] = {"type": "json_object"}
        elif fmt.kind == "json_schema":
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": fmt.schema_, "strict": True},
            }

        if request.metadata:
            params["metadata"] = dict(request.metadata)

        return params

    def _messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Render the conversation, fanning tool results out into one message each.

        The only translator in the framework that is not one-to-one. A framework
        tool turn holds every result from a round of parallel calls; OpenAI wants
        them as separate ``tool`` messages, each naming the call it answers.
        """
        messages: list[dict[str, Any]] = []

        if request.system:
            messages.append({"role": "system", "content": request.system})

        for message in request.messages:
            if message.role is Role.TOOL:
                messages.extend(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.content,
                    }
                    for result in message.tool_results
                )
                continue

            messages.append(self._turn(message))

        return messages

    def _turn(self, message: Message) -> dict[str, Any]:
        """Render one user or assistant message."""
        rendered: dict[str, Any] = {"role": message.role.value}
        if message.name:
            rendered["name"] = message.name

        if message.role is Role.ASSISTANT:
            # Assistant content is a plain string here, and `None` when the turn was
            # nothing but tool calls — which OpenAI requires, and rejects `""` for.
            rendered["content"] = message.text or None
            if message.tool_calls:
                rendered["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            # Round-trip the model's own broken JSON rather than a
                            # cleaned-up version of it: the conversation must record
                            # what the model actually said, or the correction it is
                            # about to be asked to make will not make sense to it.
                            "arguments": call.malformed_arguments
                            if call.is_malformed
                            else json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            return rendered

        content: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                if part.text:
                    content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                content.append({"type": "image_url", "image_url": {"url": self._image_url(part)}})
            # Reasoning parts are dropped: Chat Completions has nowhere to put them
            # and rejects unknown content types.

        rendered["content"] = content
        return rendered

    @staticmethod
    def _image_url(part: ImagePart) -> str:
        """Render an image as a URL — a real one, or a data URI."""
        if part.url is not None:
            return part.url

        data = base64.standard_b64encode(part.data or b"").decode("ascii")
        return f"data:{part.media_type};base64,{data}"

    @staticmethod
    def _tool(spec: ToolSpec) -> dict[str, Any]:
        """Render a tool spec as an OpenAI function definition."""
        function: dict[str, Any] = {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        }
        if spec.strict:
            function["strict"] = True
        return {"type": "function", "function": function}

    @staticmethod
    def _tool_choice(choice: ToolChoice) -> Any:
        """Render a tool choice."""
        if choice.mode == "tool":
            return {"type": "function", "function": {"name": choice.name}}
        return choice.mode  # "auto" | "none" | "required" — all three are literal.

    # -- inbound translation ------------------------------------------------ #

    def _to_response(self, completion: Any) -> LLMResponse:
        """Turn a chat completion into an :class:`~sarb.contracts.LLMResponse`."""
        builder = ResponseBuilder(self._options.model)
        finish = FinishReason.STOP

        if completion.choices:
            choice = completion.choices[0]
            finish = self._FINISH.get(choice.finish_reason or "", FinishReason.STOP)

            message = choice.message
            if message.content:
                builder.add_text(message.content)

            for index, call in enumerate(message.tool_calls or ()):
                builder.add_tool_call_delta(
                    index,
                    call_id=call.id,
                    name=call.function.name,
                    arguments_delta=call.function.arguments or "",
                )

        return builder.build(
            model=completion.model or self._options.model,
            usage=self._usage(completion.usage),
            finish_reason=finish,
            provider_metadata={"id": completion.id},
        )

    @staticmethod
    def _usage(usage: Any) -> Usage:
        """Read OpenAI's usage, including its two nested detail objects.

        OpenAI's conventions happen to be ours already: ``prompt_tokens`` counts the
        cached tokens it also reports separately, and ``completion_tokens`` counts the
        reasoning tokens. So every field here is a straight read — which is worth saying
        out loud, because the Anthropic and Gemini adapters both have to do arithmetic
        at this exact point and the absence of any here looks like an oversight.

        Cache *writes* are not reported: OpenAI's caching is automatic rather than
        requested, so there is no write to attribute. The field stays zero.
        """
        if usage is None:
            return Usage()

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)

        return Usage(
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
            cached_input_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        )

    # -- errors ------------------------------------------------------------- #

    def _fail(self, exc: BaseException) -> BaseException:
        """Translate an OpenAI exception into the framework's hierarchy."""
        if isinstance(exc, SarbError):
            return exc

        sdk = self._sdk
        model = self._options.model

        if isinstance(exc, sdk.APITimeoutError):
            return ProviderTimeoutError(
                f"OpenAI did not respond within {self._options.timeout}s",
                provider=PROVIDER,
                context={"model": model, "timeout": self._options.timeout},
                cause=exc,
            )

        if isinstance(exc, sdk.APIConnectionError):
            return LLMError(
                "Could not reach the OpenAI endpoint",
                provider=PROVIDER,
                context={"model": model, "base_url": self._options.base_url},
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
            f"OpenAI call failed: {exc}",
            provider=PROVIDER,
            context={"model": model},
            cause=exc,
        )


def build(ctx: ComponentContext) -> OpenAIProvider:
    """Build the ``openai`` LLM provider from configuration."""
    sdk = import_optional("openai", extra=EXTRA, used_by=f"LLM provider {PROVIDER!r}")
    options = parse_options(ctx.config, OpenAIOptions, component=ctx.path)

    client = sdk.AsyncOpenAI(
        api_key=options.api_key.get_secret_value() if options.api_key else None,
        organization=options.organization,
        base_url=options.base_url,
        timeout=options.timeout,
        max_retries=options.max_retries,
    )
    return OpenAIProvider(client, options)
