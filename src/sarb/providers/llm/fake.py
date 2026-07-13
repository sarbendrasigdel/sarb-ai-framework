"""A scripted model. No network, no key, no bill.

This is a first-class provider, not a test fixture that leaked into ``src``. It
exists because everything the framework is *for* — agent loops, retrieval
pipelines, workflows — is built on top of the LLM port, and none of it should need
an API key to develop against or to test.

Point ``llms.default`` at it and the whole application runs:

```yaml
llms:
  default:
    provider: fake
    options:
      responses:
        - "The capital of France is Paris."
```

It is also the only provider that can be told to fail on demand, which is the only
practical way to test the code that handles a rate limit — the alternative being to
get rate limited by a real vendor, deliberately, in CI.

Every request it receives is recorded on :attr:`FakeProvider.requests`, so a test
can assert on what an engine *sent*, not only on what it did with the answer. That
is usually the assertion that matters: prompt-assembly bugs are silent, and a test
that checks only the output passes happily while the system prompt is being dropped
on the floor.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field

from sarb.config import parse_options
from sarb.contracts import (
    FinishReason,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    Role,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCallDelta,
    Usage,
    new_id,
)
from sarb.errors import (
    AuthenticationError,
    LLMError,
    ProviderTimeoutError,
    RateLimitError,
    SarbError,
)
from sarb.providers.llm._shared import LLMOptions, ResponseBuilder

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from sarb.plugins.registry import ComponentContext

__all__ = ["FakeOptions", "FakeProvider", "ScriptedToolCall", "build"]

PROVIDER = "fake"

#: What a scripted failure should raise. Named for the *situation* rather than the
#: exception class, so config reads "the provider is rate limited" instead of
#: naming one of the framework's own types.
type Failure = Literal["rate_limit", "auth", "timeout", "error"]


class ScriptedToolCall(BaseModel):
    """A tool call the fake model should emit instead of answering.

    Attributes:
        name: The tool to call.
        arguments: The arguments to call it with.
        malformed: Raw text to emit *instead of* valid arguments, exercising the
            path where a model returns broken JSON. That path is real, it is
            recoverable, and without this it is very nearly untestable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    malformed: str | None = None


class FakeOptions(LLMOptions):
    """Configuration for the ``fake`` LLM provider.

    Attributes:
        model: A name to report. Defaulted, unlike every other provider, because
            there is no wrong model to pick here.
        responses: What to say, in order. Cycles once exhausted, so a long agent run
            does not need a long script.
        tool_calls: Tool calls to emit on the *first* turn that offers tools,
            instead of text. The next turn then answers from :attr:`responses` —
            which is exactly the call-then-answer shape an agent loop must handle,
            and the shape most agent-loop bugs live in.
        echo: Answer by echoing the last user message. For when what is under test
            is prompt assembly rather than the answer.
        chunk_size: Characters per :class:`~sarb.contracts.TextDelta` when
            streaming.
        latency: Seconds to wait before responding. For exercising timeouts and
            concurrency without a genuinely slow model.
        fail_with: Raise instead of answering.
        fail_times: How many calls fail before the script resumes. ``None`` means
            every call, forever.

    Why ``fail_times`` exists:
        "Fails forever" cannot express the case that actually matters — a call that is
        rate limited twice and then succeeds. Without it, a retry policy can only be
        tested against a provider that never recovers, which proves the policy gives up
        and proves nothing about whether it ever gets an answer.
    """

    model: str = "fake"
    responses: tuple[str, ...] = ("This is a fake response.",)
    tool_calls: tuple[ScriptedToolCall, ...] = ()
    echo: bool = False
    chunk_size: int = Field(default=8, gt=0)
    latency: float = Field(default=0.0, ge=0)
    fail_with: Failure | None = None
    fail_times: int | None = Field(default=None, ge=0)


class FakeProvider:
    """An :class:`~sarb.contracts.LLMProvider` that answers from a script."""

    def __init__(self, options: FakeOptions) -> None:
        """Load the script.

        Args:
            options: Validated configuration.
        """
        self._options = options
        self._answers = itertools.cycle(options.responses or ("",))
        self._failures = 0
        #: Every request this provider has been handed, in order. Assert on it.
        self.requests: list[LLMRequest] = []

    # -- port --------------------------------------------------------------- #

    @property
    def model(self) -> str:
        """The configured model name."""
        return self._options.model

    @property
    def capabilities(self) -> LLMCapabilities:
        """Everything, so that no engine skips a branch on account of the fake."""
        return LLMCapabilities(
            streaming=True,
            tools=True,
            vision=True,
            json_schema=True,
            reasoning=False,
            max_input_tokens=self._options.max_input_tokens,
            max_output_tokens=self._options.max_output_tokens,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Answer from the script."""
        await self._accept(request)

        builder = ResponseBuilder(self._options.model)

        if self._should_call_tools(request):
            for index, call in enumerate(self._options.tool_calls):
                # Through the delta path, not add_tool_call, so that a `malformed`
                # script goes through exactly the JSON parsing a real provider's
                # arguments would — and lands as `malformed_arguments`, not as a
                # crash inside the fake.
                builder.add_tool_call_delta(
                    index,
                    call_id=new_id(),
                    name=call.name,
                    arguments_delta=call.malformed or json.dumps(call.arguments),
                )
            return builder.build(
                usage=self._usage(request, ""), finish_reason=FinishReason.TOOL_CALLS
            )

        text = self._answer(request)
        builder.add_text(text)
        return builder.build(usage=self._usage(request, text))

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Answer from the script, a few characters at a time.

        Built on top of :meth:`generate`, so a stream and a plain call cannot
        disagree about what the script says. A fake whose two paths drift apart
        would find bugs that do not exist and hide the ones that do.
        """
        response = await self.generate(request)

        yield StreamStart(model=self._options.model)

        text = response.text
        size = self._options.chunk_size
        for start in range(0, len(text), size):
            yield TextDelta(text=text[start : start + size])

        for index, call in enumerate(response.tool_calls):
            yield ToolCallDelta(
                index=index,
                id=call.id,
                name=call.name,
                arguments_delta=call.malformed_arguments or json.dumps(call.arguments),
            )

        yield StreamEnd(response=response)

    # -- the script --------------------------------------------------------- #

    async def _accept(self, request: LLMRequest) -> None:
        """Record the request, then honour the configured latency and failure."""
        self.requests.append(request)

        if self._options.latency:
            await anyio.sleep(self._options.latency)

        if self._options.fail_with is not None and self._failing():
            raise self._failure(self._options.fail_with)

    def _failing(self) -> bool:
        """Whether this call is one of the scripted failures."""
        if self._options.fail_times is None:
            return True

        if self._failures >= self._options.fail_times:
            return False

        self._failures += 1
        return True

    def _should_call_tools(self, request: LLMRequest) -> bool:
        """Emit the scripted tool calls only on the first turn that offers tools.

        Re-emitting them once their results have come back would loop forever —
        a fine way to hang a test suite, and a poor way to model a model.
        """
        if not self._options.tool_calls or not request.tools:
            return False

        return not any(message.tool_results for message in request.messages)

    def _answer(self, request: LLMRequest) -> str:
        """The next scripted answer."""
        if self._options.echo:
            return next(
                (m.text for m in reversed(request.messages) if m.role is Role.USER),
                "",
            )

        return next(self._answers)

    @staticmethod
    def _usage(request: LLMRequest, text: str) -> Usage:
        """Plausible token counts, so budgeting code has something to divide by.

        Four characters to a token is a rule of thumb, not a tokenizer. It exists so
        that an engine which sums usage has a non-zero number to sum. Nobody should
        be billing anyone off this.
        """
        prompt = sum(len(message.text) for message in request.messages) + len(request.system or "")
        return Usage(input_tokens=prompt // 4, output_tokens=len(text) // 4)

    def _failure(self, failure: Failure) -> SarbError:
        """Build the scripted exception."""
        model = self._options.model
        context = {"model": model}

        if failure == "rate_limit":
            return RateLimitError(
                "Scripted rate limit", retry_after=1.0, provider=PROVIDER, context=context
            )
        if failure == "auth":
            return AuthenticationError(
                "Scripted authentication failure", provider=PROVIDER, context=context
            )
        if failure == "timeout":
            return ProviderTimeoutError("Scripted timeout", provider=PROVIDER, context=context)

        return LLMError("Scripted failure", provider=PROVIDER, context=context)


def build(ctx: ComponentContext) -> FakeProvider:
    """Build the ``fake`` LLM provider from configuration."""
    return FakeProvider(parse_options(ctx.config, FakeOptions, component=ctx.path))
