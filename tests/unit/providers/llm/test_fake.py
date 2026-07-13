"""Tests for the fake provider.

The fake is what every engine above this layer will be tested against, so a bug in
it becomes a bug in the test suite of everything downstream — a fake that quietly
does the wrong thing produces green tests for broken code, which is worse than
having no tests at all.
"""

from __future__ import annotations

import pytest

from sarb.contracts import (
    FinishReason,
    LLMProvider,
    LLMRequest,
    Message,
    StreamStart,
    TextDelta,
    ToolResult,
    ToolSpec,
)
from sarb.errors import AuthenticationError, LLMError, ProviderTimeoutError, RateLimitError
from sarb.providers.llm.fake import FakeOptions, FakeProvider, ScriptedToolCall

from .assertions import end_of

SEARCH = ToolSpec(
    name="search",
    description="Search the web.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)


def provider(**options) -> FakeProvider:
    return FakeProvider(FakeOptions(**options))


class TestScript:
    async def test_it_answers_from_the_script(self):
        llm = provider(responses=("Paris.",))

        assert (await llm.generate(LLMRequest(messages=(Message.user("hi"),)))).text == "Paris."

    async def test_responses_cycle(self):
        # So a long agent run does not need a long script.
        llm = provider(responses=("one", "two"))
        request = LLMRequest(messages=(Message.user("hi"),))

        answers = [(await llm.generate(request)).text for _ in range(3)]

        assert answers == ["one", "two", "one"]

    async def test_echo(self):
        llm = provider(echo=True)

        response = await llm.generate(LLMRequest(messages=(Message.user("say this back"),)))

        assert response.text == "say this back"

    async def test_it_records_what_it_was_sent(self):
        # The assertion that usually matters. Prompt-assembly bugs are silent: a test
        # that only checks the answer passes happily while the system prompt is being
        # dropped on the floor.
        llm = provider()

        await llm.generate(LLMRequest(messages=(Message.user("hi"),), system="Be terse."))

        assert llm.requests[0].system == "Be terse."


class TestToolCalls:
    async def test_it_calls_the_scripted_tool_then_answers(self):
        # The call-then-answer shape an agent loop must handle, and where most
        # agent-loop bugs live.
        llm = provider(
            responses=("Paris.",),
            tool_calls=(ScriptedToolCall(name="search", arguments={"q": "capital"}),),
        )

        first = await llm.generate(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))

        assert first.finish_reason is FinishReason.TOOL_CALLS
        assert first.tool_calls[0].name == "search"
        assert first.tool_calls[0].arguments == {"q": "capital"}

        second = await llm.generate(
            LLMRequest(
                messages=(
                    Message.user("hi"),
                    Message.assistant(tool_calls=first.tool_calls),
                    Message.tool(
                        [ToolResult(call_id=first.tool_calls[0].id, name="search", content="Paris")]
                    ),
                ),
                tools=(SEARCH,),
            )
        )

        # Re-emitting the calls once their results are back would loop forever.
        assert second.finish_reason is FinishReason.STOP
        assert second.text == "Paris."

    async def test_it_can_be_told_to_emit_broken_json(self):
        # A real and recoverable event that is otherwise very nearly untestable.
        llm = provider(tool_calls=(ScriptedToolCall(name="search", malformed='{"q":'),))

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))

        call = response.tool_calls[0]
        assert call.is_malformed
        assert call.malformed_arguments == '{"q":'
        assert call.arguments == {}

    async def test_no_tools_offered_means_no_tools_called(self):
        llm = provider(responses=("Paris.",), tool_calls=(ScriptedToolCall(name="search"),))

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert response.tool_calls == ()
        assert response.text == "Paris."


class TestStreaming:
    async def test_it_streams_in_chunks_and_ends_assembled(self):
        llm = provider(responses=("Paris is the capital.",), chunk_size=6)

        events = [e async for e in llm.stream(LLMRequest(messages=(Message.user("hi"),)))]

        assert isinstance(events[0], StreamStart)
        assert [e.text for e in events if isinstance(e, TextDelta)] == [
            "Paris ",
            "is the",
            " capit",
            "al.",
        ]

        response = end_of(events)
        assert response.text == "Paris is the capital."

    async def test_streaming_and_generating_cannot_disagree(self):
        # Built on top of generate() for exactly this reason. A fake whose two paths
        # drift apart finds bugs that do not exist and hides the ones that do.
        llm = provider(responses=("Paris.",))
        request = LLMRequest(messages=(Message.user("hi"),))

        streamed = end_of([e async for e in llm.stream(request)])
        generated = await llm.generate(request)

        assert streamed.text == generated.text


class TestFailure:
    @pytest.mark.parametrize(
        ("failure", "expected"),
        [
            ("rate_limit", RateLimitError),
            ("auth", AuthenticationError),
            ("timeout", ProviderTimeoutError),
            ("error", LLMError),
        ],
    )
    async def test_it_can_be_told_to_fail(self, failure, expected):
        # The only honest way to test a retry policy. The alternative is getting rate
        # limited by a real vendor, deliberately, in CI.
        llm = provider(fail_with=failure)

        with pytest.raises(expected):
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

    async def test_a_scripted_rate_limit_is_retryable_and_says_how_long(self):
        llm = provider(fail_with="rate_limit")

        with pytest.raises(RateLimitError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert caught.value.retryable
        assert caught.value.retry_after == 1.0

    async def test_a_failed_call_is_still_recorded(self):
        llm = provider(fail_with="error")

        with pytest.raises(LLMError):
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert len(llm.requests) == 1


class TestPort:
    def test_it_satisfies_the_port_by_shape(self):
        assert isinstance(provider(), LLMProvider)

    def test_it_claims_every_capability(self):
        # So that no engine skips a branch on account of the fake.
        caps = provider().capabilities

        assert caps.streaming
        assert caps.tools
        assert caps.vision
        assert caps.json_schema

    async def test_usage_is_non_zero_so_budgeting_code_has_something_to_sum(self):
        llm = provider(responses=("a much longer answer than the question",))

        usage = (await llm.generate(LLMRequest(messages=(Message.user("hi there"),)))).usage

        assert usage.total_tokens > 0
