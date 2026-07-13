"""Gemini, through Google's Gen AI SDK.

Gemini's shape is further from the framework's than either Anthropic's or
OpenAI's, and this is where the value of having a neutral model is easiest to see.
Four disagreements, and what is done about each:

**The assistant is called ``model``.** A one-word rename, and the kind of thing
that would otherwise leak into an engine as ``if provider == "gemini"``.

**Tool results are content parts, not a role.** A ``function_response`` part inside
a *user* turn. The framework's tool turn maps onto it directly.

**Tool calls do not always carry an id.** Gemini matches a response to a call by
tool *name*, which is ambiguous the moment a model calls the same tool twice in one
turn — which they do. The adapter mints an id when Gemini omits one and echoes it
back on both the call and the result, so the history it sends is internally
consistent and parallel calls stay distinguishable.

**Reasoning is a flag on a text part.** ``part.thought`` is a boolean, so a
reasoning part and an answer part are the same type wearing different hats. Getting
this wrong means streaming the model's scratch work into the answer box, which is
exactly the mistake :class:`~sarb.contracts.ReasoningDelta` exists to make you opt
into.
"""

from __future__ import annotations

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
    ReasoningDelta,
    Role,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextPart,
    ToolCallDelta,
    ToolChoice,
    Usage,
    new_id,
)
from sarb.dependencies import import_optional
from sarb.errors import LLMError, SarbError
from sarb.providers._shared import error_for_status
from sarb.providers.llm._shared import LLMOptions, ResponseBuilder

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from google import genai

    from sarb.plugins.registry import ComponentContext

__all__ = ["GeminiOptions", "GeminiProvider", "build"]

PROVIDER = "gemini"
EXTRA = "gemini"


class GeminiOptions(LLMOptions):
    """Configuration for the ``gemini`` LLM provider.

    Attributes:
        api_key: The API key. Omit it and the SDK reads ``GOOGLE_API_KEY`` or
            ``GEMINI_API_KEY``.
        vertex: Authenticate against Vertex AI using application-default
            credentials instead of an API key. ``project`` and ``location`` are
            then required.
        project: GCP project id. Vertex only.
        location: GCP region, e.g. ``us-central1``. Vertex only.
        thinking: Return the model's reasoning as well as its answer.
        thinking_level: How much reasoning to do, where the model supports it.

    Example:
        ```yaml
        llms:
          default:
            provider: gemini
            options:
              model: gemini-2.0-flash
              api_key: ${GEMINI_API_KEY}
        ```
    """

    api_key: SecretStr | None = None
    vertex: bool = False
    project: str | None = None
    location: str | None = None
    thinking: bool = False
    thinking_level: Literal["low", "high"] | None = None

    @property
    def timeout_ms(self) -> int:
        """The timeout, in the milliseconds this SDK wants."""
        return int(self.timeout * 1000)


class GeminiProvider:
    """An :class:`~sarb.contracts.LLMProvider` backed by Gemini.

    Use :func:`build` to construct one from configuration.
    """

    _FINISH: ClassVar[dict[str, FinishReason]] = {
        "STOP": FinishReason.STOP,
        "MAX_TOKENS": FinishReason.LENGTH,
        "SAFETY": FinishReason.CONTENT_FILTER,
        "RECITATION": FinishReason.CONTENT_FILTER,
        "BLOCKLIST": FinishReason.CONTENT_FILTER,
        "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
        "SPII": FinishReason.CONTENT_FILTER,
        "MALFORMED_FUNCTION_CALL": FinishReason.ERROR,
    }

    _TOOL_CHOICE: ClassVar[dict[str, str]] = {
        "auto": "AUTO",
        "none": "NONE",
        "required": "ANY",
        "tool": "ANY",  # narrowed further by allowed_function_names
    }

    def __init__(self, client: genai.Client, options: GeminiOptions) -> None:
        """Wrap a Gen AI client.

        Args:
            client: A ``google.genai.Client``.
            options: Validated configuration.
        """
        self._client = client
        self._options = options
        self._genai = import_optional(
            "google.genai", extra=EXTRA, used_by=f"LLM provider {PROVIDER!r}"
        )
        self._types = self._genai.types
        self._errors = self._genai.errors

    # -- port --------------------------------------------------------------- #

    @property
    def model(self) -> str:
        """The configured model."""
        return self._options.model

    @property
    def capabilities(self) -> LLMCapabilities:
        """What the Gen AI API supports."""
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
        contents, config = self._params(request)
        try:
            response = await self._client.aio.models.generate_content(
                model=self._options.model, contents=contents, config=config
            )
        except Exception as exc:
            raise self._fail(exc) from exc

        builder = ResponseBuilder(self._options.model)
        finish = self._consume(response, builder)
        return builder.build(
            model=getattr(response, "model_version", None) or self._options.model,
            usage=self._usage(response.usage_metadata),
            finish_reason=finish,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Call the model and yield events as they arrive."""
        contents, config = self._params(request)
        builder = ResponseBuilder(self._options.model)
        model = self._options.model
        usage = Usage()
        finish = FinishReason.STOP
        started = False
        calls = 0

        try:
            chunks = await self._client.aio.models.generate_content_stream(
                model=model, contents=contents, config=config
            )
            async for chunk in chunks:
                if not started:
                    model = getattr(chunk, "model_version", None) or model
                    started = True
                    yield StreamStart(model=model)

                if chunk.usage_metadata is not None:
                    usage = self._usage(chunk.usage_metadata)

                for part, is_call in self._parts(chunk):
                    if is_call:
                        call = part.function_call
                        call_id = call.id or new_id()
                        builder.add_tool_call(
                            call_id=call_id, name=call.name or "", arguments=dict(call.args or {})
                        )
                        yield ToolCallDelta(index=calls, id=call_id, name=call.name)
                        calls += 1
                    elif part.thought:
                        builder.add_reasoning(part.text or "")
                        yield ReasoningDelta(text=part.text or "")
                    elif part.text:
                        builder.add_text(part.text)
                        yield TextDelta(text=part.text)

                reason = self._finish_reason(chunk)
                if reason is not None:
                    finish = reason
        except Exception as exc:
            raise self._fail(exc) from exc

        if not started:
            yield StreamStart(model=model)

        # Gemini says STOP even when it has asked for tools. Correcting that here
        # is not cosmetic: the agent loop keys on TOOL_CALLS to decide whether the
        # turn is over, and would otherwise stop with the tools uncalled.
        if calls and finish is FinishReason.STOP:
            finish = FinishReason.TOOL_CALLS

        yield StreamEnd(response=builder.build(model=model, usage=usage, finish_reason=finish))

    # -- outbound translation ----------------------------------------------- #

    def _params(self, request: LLMRequest) -> tuple[list[Any], Any]:
        """Turn an :class:`~sarb.contracts.LLMRequest` into contents plus config."""
        types = self._types
        options = self._options

        config: dict[str, Any] = {}
        if request.system:
            config["system_instruction"] = request.system
        if request.max_tokens is not None:
            config["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["top_p"] = request.top_p
        if request.stop:
            config["stop_sequences"] = list(request.stop)
        if request.seed is not None:
            config["seed"] = request.seed

        if request.tools:
            config["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=spec.name,
                            description=spec.description,
                            # `parameters_json_schema` takes plain JSON Schema.
                            # `parameters` takes Gemini's own Schema type, which
                            # would mean maintaining a converter for a dialect
                            # nobody else speaks.
                            parameters_json_schema=spec.parameters,
                        )
                        for spec in request.tools
                    ]
                )
            ]
            config["tool_config"] = self._tool_choice(request.tool_choice)
            # The SDK will happily *execute* Python callables it recognises. We pass
            # declarations, so it has nothing to run — but turning the behaviour off
            # explicitly means a future SDK default cannot start calling tools
            # behind the agent loop's back.
            config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )

        fmt = request.response_format
        if fmt.kind == "json":
            config["response_mime_type"] = "application/json"
        elif fmt.kind == "json_schema":
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = fmt.schema_

        if options.thinking:
            thinking: dict[str, Any] = {"include_thoughts": True}
            if options.thinking_level:
                thinking["thinking_level"] = options.thinking_level.upper()
            config["thinking_config"] = types.ThinkingConfig(**thinking)

        contents = [self._content(message) for message in request.messages]
        return contents, types.GenerateContentConfig(**config)

    def _content(self, message: Message) -> Any:
        """Render one framework message as a Gemini ``Content``."""
        types = self._types

        if message.role is Role.TOOL:
            return types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=result.call_id,
                            name=result.name,
                            # Gemini wants a JSON object, not a string. A tool that
                            # failed says so under `error`, which is the convention
                            # the model is trained on.
                            response={"error": result.content}
                            if result.is_error
                            else {"result": result.content},
                        )
                    )
                    for result in message.tool_results
                ],
            )

        parts: list[Any] = []
        for part in message.content:
            if isinstance(part, TextPart):
                if part.text:
                    parts.append(types.Part(text=part.text))
            elif isinstance(part, ImagePart):
                parts.append(self._image_part(part))
            # Reasoning is not echoed back: Gemini regenerates it, and a `thought`
            # part sent from the client is not part of the API.

        for call in message.tool_calls:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=call.id, name=call.name, args=call.arguments
                    )
                )
            )

        role = "model" if message.role is Role.ASSISTANT else "user"
        return types.Content(role=role, parts=parts)

    def _image_part(self, part: ImagePart) -> Any:
        """Render an image as a Gemini part."""
        types = self._types
        if part.url is not None:
            return types.Part(
                file_data=types.FileData(file_uri=part.url, mime_type=part.media_type)
            )
        return types.Part(inline_data=types.Blob(data=part.data or b"", mime_type=part.media_type))

    def _tool_choice(self, choice: ToolChoice) -> Any:
        """Render a tool choice as a ``ToolConfig``."""
        types = self._types
        config: dict[str, Any] = {"mode": self._TOOL_CHOICE[choice.mode]}
        if choice.mode == "tool" and choice.name:
            config["allowed_function_names"] = [choice.name]

        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(**config))

    # -- inbound translation ------------------------------------------------ #

    def _consume(self, response: Any, builder: ResponseBuilder) -> FinishReason:
        """Read every part of a complete response into the builder."""
        calls = 0
        for part, is_call in self._parts(response):
            if is_call:
                call = part.function_call
                builder.add_tool_call(
                    call_id=call.id or new_id(),
                    name=call.name or "",
                    arguments=dict(call.args or {}),
                )
                calls += 1
            elif part.thought:
                builder.add_reasoning(part.text or "")
            elif part.text:
                builder.add_text(part.text)

        finish = self._finish_reason(response) or FinishReason.STOP
        if calls and finish is FinishReason.STOP:
            return FinishReason.TOOL_CALLS
        return finish

    @staticmethod
    def _parts(response: Any) -> list[tuple[Any, bool]]:
        """Every content part of the first candidate, flagged as a call or not."""
        candidates = response.candidates or ()
        if not candidates:
            return []

        content = candidates[0].content
        if content is None or not content.parts:
            return []

        return [(part, part.function_call is not None) for part in content.parts]

    def _finish_reason(self, response: Any) -> FinishReason | None:
        """Read the finish reason, if this chunk carried one."""
        candidates = response.candidates or ()
        if not candidates:
            return None

        reason = candidates[0].finish_reason
        if reason is None:
            return None

        return self._FINISH.get(str(reason.value if hasattr(reason, "value") else reason))

    @staticmethod
    def _usage(usage: Any) -> Usage:
        """Read Gemini's usage metadata.

        **Gemini's thinking tokens are not inside its output tokens.** Google's own
        pricing wording is that *"response pricing is the sum of output tokens and
        thinking tokens"* — they are billed as output and reported beside it, where
        OpenAI and Anthropic report theirs within it. Our contract says ``output_tokens``
        is everything generated, so they are added here.

        Miss this and every reasoning call under-reports output by exactly the part that
        is most expensive, which is the error you would least like to make.
        """
        if usage is None:
            return Usage()

        answer = usage.candidates_token_count or 0
        thoughts = usage.thoughts_token_count or 0

        return Usage(
            # `prompt_token_count` already counts the cached prefix, so this one is a
            # straight read — the convention happens to match ours.
            input_tokens=usage.prompt_token_count or 0,
            output_tokens=answer + thoughts,
            cached_input_tokens=usage.cached_content_token_count or 0,
            reasoning_tokens=thoughts,
        )

    # -- errors ------------------------------------------------------------- #

    def _fail(self, exc: BaseException) -> BaseException:
        """Translate a Gen AI exception into the framework's hierarchy."""
        if isinstance(exc, SarbError):
            return exc

        model = self._options.model
        if isinstance(exc, self._errors.APIError):
            return error_for_status(
                exc.code,
                exc.message or str(exc),
                provider=PROVIDER,
                model=model,
                otherwise=LLMError,
                cause=exc,
            )

        return LLMError(
            f"Gemini call failed: {exc}",
            provider=PROVIDER,
            context={"model": model},
            cause=exc,
        )


def build(ctx: ComponentContext) -> GeminiProvider:
    """Build the ``gemini`` LLM provider from configuration."""
    genai_module = import_optional(
        "google.genai", extra=EXTRA, used_by=f"LLM provider {PROVIDER!r}"
    )
    options = parse_options(ctx.config, GeminiOptions, component=ctx.path)

    http_options = genai_module.types.HttpOptions(
        base_url=options.base_url,
        timeout=options.timeout_ms,
    )

    if options.vertex:
        client = genai_module.Client(
            vertexai=True,
            project=options.project,
            location=options.location,
            http_options=http_options,
        )
    else:
        client = genai_module.Client(
            api_key=options.api_key.get_secret_value() if options.api_key else None,
            http_options=http_options,
        )

    return GeminiProvider(client, options)
