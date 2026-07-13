"""Ollama, over its native HTTP API.

No SDK. Ollama's ``/api/chat`` is a small, stable JSON endpoint, and a vendor
client for it would be a dependency that adds a version to pin and takes away
nothing to worry about. ``httpx`` is already the transport under both other hosted
adapters, so this costs nothing extra to install.

Ollama also speaks an OpenAI-compatible dialect, which means the
:mod:`~sarb.providers.llm.openai` adapter can drive it too. Prefer *this* one: the
compatibility layer does not expose ``think``, and its tool support has historically
lagged the native API's. Reach for the OpenAI adapter against Ollama only when you
want one code path across a hosted model and a local one.

The one real wrinkle
--------------------
**Ollama's tool calls have no ids.** It matches a result to a call by tool *name*,
which is ambiguous the instant a model calls the same tool twice in one turn. The
adapter mints an id on the way in — so the framework's history stays honest and the
agent loop can pair calls with results — and on the way out sends ``tool_name``,
which is what Ollama actually reads. The id is ours; Ollama never sees it.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

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
    ToolSpec,
    Usage,
    new_id,
)
from sarb.dependencies import import_optional
from sarb.errors import LLMError, ProviderTimeoutError, SarbError
from sarb.providers._shared import error_for_status, retry_after_seconds, unsupported
from sarb.providers.llm._shared import LLMOptions, ResponseBuilder

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    import httpx

    from sarb.plugins.registry import ComponentContext

__all__ = ["OllamaOptions", "OllamaProvider", "build"]

PROVIDER = "ollama"
EXTRA = "ollama"

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaOptions(LLMOptions):
    """Configuration for the ``ollama`` LLM provider.

    Attributes:
        base_url: Where Ollama is listening. Defaults to the local daemon.
        think: Ask the model to expose its reasoning. Only some models can; the
            rest ignore it.
        keep_alive: How long Ollama should hold the model in memory after the call,
            e.g. ``"5m"`` or ``"0"`` to unload immediately. The difference between a
            200ms second request and a 20-second one.
        vision: Whether the configured model accepts images. Declared, not guessed —
            Ollama serves whatever you have pulled.
        extra_options: Passed straight through in Ollama's ``options`` block
            (``num_ctx``, ``repeat_penalty``, ``mirostat``…). An escape hatch for
            knobs the framework has no neutral concept for, and deliberately
            *not* modelled: the day the framework grows a ``mirostat`` field is the
            day it has an opinion about one vendor's sampler.

    Example:
        ```yaml
        llms:
          local:
            provider: ollama
            options:
              model: llama3.1:8b
              keep_alive: 30m
              extra_options:
                num_ctx: 8192
        ```
    """

    base_url: str = DEFAULT_BASE_URL
    think: bool = False
    keep_alive: str | None = None
    vision: bool = False
    extra_options: dict[str, Any] = Field(default_factory=dict)


class OllamaProvider:
    """An :class:`~sarb.contracts.LLMProvider` backed by a local Ollama daemon.

    Use :func:`build` to construct one from configuration.
    """

    _FINISH: ClassVar[dict[str, FinishReason]] = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "load": FinishReason.ERROR,
    }

    def __init__(self, client: httpx.AsyncClient, options: OllamaOptions) -> None:
        """Wrap an HTTP client pointed at an Ollama daemon.

        Args:
            client: An ``httpx.AsyncClient`` with ``base_url`` already set.
            options: Validated configuration.
        """
        self._client = client
        self._options = options
        self._httpx = import_optional("httpx", extra=EXTRA, used_by=f"LLM provider {PROVIDER!r}")

    # -- port --------------------------------------------------------------- #

    @property
    def model(self) -> str:
        """The configured model."""
        return self._options.model

    @property
    def capabilities(self) -> LLMCapabilities:
        """What Ollama's chat API supports."""
        return LLMCapabilities(
            streaming=True,
            tools=True,
            vision=self._options.vision,
            json_schema=True,
            reasoning=self._options.think,
            max_input_tokens=self._options.max_input_tokens,
            max_output_tokens=self._options.max_output_tokens,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Call the model and return the whole response."""
        payload = self._payload(request, stream=False)
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise self._fail(exc) from exc

        builder = ResponseBuilder(self._options.model)
        self._consume(body.get("message") or {}, builder)
        return builder.build(
            model=body.get("model") or self._options.model,
            usage=self._usage(body),
            finish_reason=self._finish(body),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Call the model and yield events as they arrive.

        Ollama streams newline-delimited JSON, one object per line, each shaped like
        a smaller version of the complete response.
        """
        payload = self._payload(request, stream=True)
        builder = ResponseBuilder(self._options.model)
        model = self._options.model
        usage = Usage()
        finish = FinishReason.STOP
        started = False
        calls = 0

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    chunk = json.loads(line)
                    if not started:
                        model = chunk.get("model") or model
                        started = True
                        yield StreamStart(model=model)

                    message = chunk.get("message") or {}

                    thinking = message.get("thinking")
                    if thinking:
                        builder.add_reasoning(thinking)
                        yield ReasoningDelta(text=thinking)

                    content = message.get("content")
                    if content:
                        builder.add_text(content)
                        yield TextDelta(text=content)

                    for call in message.get("tool_calls") or ():
                        function = call.get("function") or {}
                        call_id = new_id()
                        name = function.get("name") or ""
                        # Ollama sends whole calls, not partial JSON — there is
                        # nothing to accumulate, so the arguments go in parsed.
                        builder.add_tool_call(
                            call_id=call_id,
                            name=name,
                            arguments=dict(function.get("arguments") or {}),
                        )
                        yield ToolCallDelta(
                            index=calls,
                            id=call_id,
                            name=name,
                            arguments_delta=json.dumps(function.get("arguments") or {}),
                        )
                        calls += 1

                    if chunk.get("done"):
                        usage = self._usage(chunk)
                        finish = self._FINISH.get(
                            chunk.get("done_reason") or "stop", FinishReason.STOP
                        )
        except Exception as exc:
            raise self._fail(exc) from exc

        if not started:
            yield StreamStart(model=model)

        if calls and finish is FinishReason.STOP:
            finish = FinishReason.TOOL_CALLS

        yield StreamEnd(response=builder.build(model=model, usage=usage, finish_reason=finish))

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Called by the container on shutdown."""
        await self._client.aclose()

    # -- outbound translation ----------------------------------------------- #

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        """Turn an :class:`~sarb.contracts.LLMRequest` into an ``/api/chat`` body."""
        options = self._options

        sampling: dict[str, Any] = dict(options.extra_options)
        if request.temperature is not None:
            sampling["temperature"] = request.temperature
        if request.top_p is not None:
            sampling["top_p"] = request.top_p
        if request.max_tokens is not None:
            sampling["num_predict"] = request.max_tokens
        if request.stop:
            sampling["stop"] = list(request.stop)
        if request.seed is not None:
            sampling["seed"] = request.seed

        payload: dict[str, Any] = {
            "model": options.model,
            "messages": self._messages(request),
            "stream": stream,
        }
        if sampling:
            payload["options"] = sampling
        if options.think:
            payload["think"] = True
        if options.keep_alive is not None:
            payload["keep_alive"] = options.keep_alive

        if request.tools:
            payload["tools"] = [self._tool(tool) for tool in request.tools]
            if request.tool_choice.mode != "auto":
                # Ollama has no tool_choice. Pretending otherwise would mean a
                # `required` that silently does not require anything — and an agent
                # loop waiting for a call that is never coming.
                raise unsupported(
                    PROVIDER,
                    f'tool_choice mode "{request.tool_choice.mode}"',
                    hint="Ollama supports only automatic tool choice.",
                )

        fmt = request.response_format
        if fmt.kind == "json":
            payload["format"] = "json"
        elif fmt.kind == "json_schema":
            payload["format"] = fmt.schema_

        return payload

    def _messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Render the conversation, fanning tool results out into one message each."""
        messages: list[dict[str, Any]] = []

        if request.system:
            messages.append({"role": "system", "content": request.system})

        for message in request.messages:
            if message.role is Role.TOOL:
                messages.extend(
                    # `tool_name`, not `tool_call_id`: Ollama pairs results to calls
                    # by name. Our id would mean nothing to it.
                    {"role": "tool", "tool_name": result.name, "content": result.content}
                    for result in message.tool_results
                )
                continue

            messages.append(self._turn(message))

        return messages

    @staticmethod
    def _turn(message: Message) -> dict[str, Any]:
        """Render one user or assistant message."""
        rendered: dict[str, Any] = {
            "role": message.role.value,
            "content": "".join(part.text for part in message.content if isinstance(part, TextPart)),
        }

        images = [
            base64.standard_b64encode(part.data).decode("ascii")
            for part in message.content
            if isinstance(part, ImagePart) and part.data is not None
        ]
        if images:
            # Ollama takes base64 bytes only; it will not fetch a URL for you.
            rendered["images"] = images

        if message.tool_calls:
            rendered["tool_calls"] = [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in message.tool_calls
            ]

        return rendered

    @staticmethod
    def _tool(spec: ToolSpec) -> dict[str, Any]:
        """Render a tool spec as an Ollama tool definition."""
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }

    # -- inbound translation ------------------------------------------------ #

    @staticmethod
    def _consume(message: dict[str, Any], builder: ResponseBuilder) -> None:
        """Read a complete Ollama message into the builder."""
        if message.get("thinking"):
            builder.add_reasoning(message["thinking"])
        if message.get("content"):
            builder.add_text(message["content"])

        for call in message.get("tool_calls") or ():
            function = call.get("function") or {}
            builder.add_tool_call(
                call_id=new_id(),
                name=function.get("name") or "",
                arguments=dict(function.get("arguments") or {}),
            )

    def _finish(self, body: dict[str, Any]) -> FinishReason:
        """Decide why generation stopped."""
        reason = self._FINISH.get(body.get("done_reason") or "stop", FinishReason.STOP)

        message = body.get("message") or {}
        if message.get("tool_calls") and reason is FinishReason.STOP:
            # As with Gemini: Ollama reports a clean stop even when it is waiting for
            # tools, and an agent loop reading STOP would end the turn there.
            return FinishReason.TOOL_CALLS

        return reason

    @staticmethod
    def _usage(body: dict[str, Any]) -> Usage:
        """Read Ollama's token counts."""
        return Usage(
            input_tokens=body.get("prompt_eval_count") or 0,
            output_tokens=body.get("eval_count") or 0,
        )

    # -- errors ------------------------------------------------------------- #

    def _fail(self, exc: BaseException) -> BaseException:
        """Translate an HTTP failure into the framework's hierarchy."""
        if isinstance(exc, SarbError):
            return exc

        httpx_module = self._httpx
        model = self._options.model

        if isinstance(exc, httpx_module.TimeoutException):
            return ProviderTimeoutError(
                f"Ollama did not respond within {self._options.timeout}s",
                provider=PROVIDER,
                context={"model": model, "timeout": self._options.timeout},
                cause=exc,
            )

        if isinstance(exc, httpx_module.HTTPStatusError):
            return error_for_status(
                exc.response.status_code,
                self._detail(exc.response),
                provider=PROVIDER,
                model=model,
                otherwise=LLMError,
                retry_after=retry_after_seconds(exc.response.headers),
                cause=exc,
            )

        if isinstance(exc, httpx_module.TransportError):
            return LLMError(
                f"Could not reach Ollama at {self._options.base_url}. Is it running?",
                provider=PROVIDER,
                context={"model": model, "base_url": self._options.base_url},
                cause=exc,
                retryable=True,
            )

        return LLMError(
            f"Ollama call failed: {exc}",
            provider=PROVIDER,
            context={"model": model},
            cause=exc,
        )

    @staticmethod
    def _detail(response: Any) -> str:
        """Ollama's own error text, which is more useful than the status line.

        A 404 from Ollama almost always means the model has not been pulled, and it
        says so — but only in the body.
        """
        try:
            body = response.json()
        except ValueError:
            return response.text or f"Ollama returned {response.status_code}"

        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])
        return f"Ollama returned {response.status_code}"


def build(ctx: ComponentContext) -> OllamaProvider:
    """Build the ``ollama`` LLM provider from configuration."""
    httpx_module = import_optional("httpx", extra=EXTRA, used_by=f"LLM provider {PROVIDER!r}")
    options = parse_options(ctx.config, OllamaOptions, component=ctx.path)

    client = httpx_module.AsyncClient(
        base_url=options.base_url.rstrip("/"),
        timeout=options.timeout,
    )
    return OllamaProvider(client, options)
