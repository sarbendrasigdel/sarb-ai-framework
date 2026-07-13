"""Tests for the OpenAI adapter.

As with Anthropic, fixtures are built from OpenAI's own Pydantic models, so the
SDK validates the shape we claim to receive.

The assertion that earns its keep here is the tool-result fan-out. It is the only
place in the framework where a translator is not one-to-one, which makes it the
only place where a message can be *lost* rather than merely mistranslated.
"""

from __future__ import annotations

import pytest

openai = pytest.importorskip("openai")

import httpx
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)

from sarb.contracts import (
    FinishReason,
    ImagePart,
    LLMProvider,
    LLMRequest,
    Message,
    ResponseFormat,
    Role,
    TextDelta,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolSpec,
)
from sarb.errors import AuthenticationError, LLMError, RateLimitError
from sarb.providers.llm.openai import OpenAIOptions, OpenAIProvider

from .assertions import end_of

SEARCH = ToolSpec(
    name="search",
    description="Search the web.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)


class FakeCompletions:
    def __init__(self, result):
        self._result = result
        self.params: dict[str, object] = {}

    async def create(self, **params):
        self.params = params
        if isinstance(self._result, BaseException):
            raise self._result
        if params.get("stream"):
            return _aiter(self._result)
        return self._result


class FakeClient:
    def __init__(self, result=None):
        self.chat = type("Chat", (), {"completions": FakeCompletions(result)})()
        self.closed = False

    async def close(self):
        self.closed = True


async def _aiter(chunks):
    for chunk in chunks:
        yield chunk


def provider(result=None, **options) -> OpenAIProvider:
    return OpenAIProvider(
        FakeClient(result),  # type: ignore[arg-type]
        OpenAIOptions(model="gpt-4o", **options),
    )


def sent(llm):
    """The kwargs the adapter would actually have sent to OpenAI."""
    return llm._client.chat.completions.params


def reply(
    content: str | None = "Paris.",
    *,
    tool_calls=None,
    finish_reason="stop",
    cached: int = 0,
    reasoning: int = 0,
) -> ChatCompletion:
    return ChatCompletion(
        id="cmpl_1",
        model="gpt-4o",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                finish_reason=finish_reason,
                message=ChatCompletionMessage(
                    role="assistant", content=content, tool_calls=tool_calls
                ),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=reasoning),
        ),
    )


def call(id_: str, name: str, arguments: str) -> ChatCompletionMessageFunctionToolCall:
    return ChatCompletionMessageFunctionToolCall(
        id=id_, type="function", function=Function(name=name, arguments=arguments)
    )


class TestRequestTranslation:
    async def test_the_system_prompt_becomes_the_first_message(self):
        llm = provider(reply())

        await llm.generate(LLMRequest(messages=(Message.user("hi"),), system="Be terse."))

        assert sent(llm)["messages"][0] == {"role": "system", "content": "Be terse."}

    async def test_one_tool_turn_fans_out_into_one_message_per_result(self):
        # The only non-one-to-one translation in the framework. The neutral model
        # keeps one model turn and one response turn; OpenAI's shape is produced
        # here, where it belongs, and nowhere else.
        llm = provider(reply("done"))

        await llm.generate(
            LLMRequest(
                messages=(
                    Message.user("search"),
                    Message.assistant(tool_calls=(ToolCall(id="c1", name="search"),)),
                    Message.tool(
                        [
                            ToolResult(call_id="c1", name="search", content="Paris"),
                            ToolResult(call_id="c2", name="search", content="Lyon"),
                        ]
                    ),
                ),
                tools=(SEARCH,),
            )
        )

        messages = sent(llm)["messages"]
        assert messages[-2] == {"role": "tool", "tool_call_id": "c1", "content": "Paris"}
        assert messages[-1] == {"role": "tool", "tool_call_id": "c2", "content": "Lyon"}

    async def test_an_assistant_turn_of_only_tool_calls_has_null_content(self):
        # OpenAI requires null here and rejects "".
        llm = provider(reply("done"))

        await llm.generate(
            LLMRequest(
                messages=(
                    Message.user("search"),
                    Message.assistant(
                        tool_calls=(ToolCall(id="c1", name="search", arguments={"q": "paris"}),)
                    ),
                ),
                tools=(SEARCH,),
            )
        )

        assistant = sent(llm)["messages"][1]
        assert assistant["content"] is None
        assert assistant["tool_calls"][0]["function"] == {
            "name": "search",
            "arguments": '{"q": "paris"}',
        }

    async def test_the_models_own_broken_json_is_echoed_back_unrepaired(self):
        # The conversation must record what the model actually said. Sending a
        # cleaned-up version means the correction it is about to be asked to make
        # will not make sense to it.
        llm = provider(reply("done"))

        await llm.generate(
            LLMRequest(
                messages=(
                    Message.user("search"),
                    Message.assistant(
                        tool_calls=(ToolCall(id="c1", name="search", malformed_arguments='{"q":'),)
                    ),
                ),
                tools=(SEARCH,),
            )
        )

        assert sent(llm)["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"q":'

    async def test_an_inline_image_becomes_a_data_uri(self):
        llm = provider(reply())

        await llm.generate(
            LLMRequest(
                messages=(
                    Message(
                        role=Role.USER,
                        content=(ImagePart(data=b"\x89PNG", media_type="image/png"),),
                    ),
                )
            )
        )

        part = sent(llm)["messages"][0]["content"][0]
        assert part["image_url"]["url"] == "data:image/png;base64,iVBORw=="

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            (ToolChoice.auto(), "auto"),
            (ToolChoice.none(), "none"),
            (ToolChoice.required(), "required"),
            (ToolChoice.tool("search"), {"type": "function", "function": {"name": "search"}}),
        ],
    )
    async def test_tool_choice(self, choice, expected):
        llm = provider(reply())

        await llm.generate(
            LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,), tool_choice=choice)
        )

        assert sent(llm)["tool_choice"] == expected

    async def test_the_output_cap_field_is_configurable_for_compatible_servers(self):
        # `max_completion_tokens` is what OpenAI wants; most compatible servers only
        # implement `max_tokens`. Guessing wrong silently drops the cap.
        llm = provider(reply(), token_limit_field="max_tokens")

        await llm.generate(LLMRequest(messages=(Message.user("hi"),), max_tokens=256))

        assert sent(llm)["max_tokens"] == 256
        assert "max_completion_tokens" not in sent(llm)

    async def test_json_schema_output(self):
        llm = provider(reply("{}"))
        schema = {"type": "object", "properties": {"city": {"type": "string"}}}

        await llm.generate(
            LLMRequest(
                messages=(Message.user("hi"),),
                response_format=ResponseFormat(kind="json_schema", schema_=schema),
            )
        )

        assert sent(llm)["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }


class TestResponseTranslation:
    async def test_text(self):
        response = await provider(reply()).generate(LLMRequest(messages=(Message.user("hi"),)))

        assert response.text == "Paris."
        assert response.finish_reason is FinishReason.STOP

    async def test_tool_calls(self):
        llm = provider(
            reply(
                "Looking.",
                tool_calls=[call("c1", "search", '{"q": "paris"}')],
                finish_reason="tool_calls",
            )
        )

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))

        assert response.finish_reason is FinishReason.TOOL_CALLS
        assert response.tool_calls[0].arguments == {"q": "paris"}

    async def test_broken_arguments_are_carried_not_raised(self):
        llm = provider(
            reply(None, tool_calls=[call("c1", "search", '{"q":')], finish_reason="tool_calls")
        )

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))

        assert response.tool_calls[0].is_malformed
        assert response.tool_calls[0].malformed_arguments == '{"q":'

    async def test_usage_includes_cached_and_reasoning_tokens(self):
        llm = provider(reply(cached=3, reasoning=2))

        usage = (await llm.generate(LLMRequest(messages=(Message.user("hi"),)))).usage

        assert usage.cached_input_tokens == 3
        assert usage.reasoning_tokens == 2

    def test_reasoning_is_billed_but_not_returned_so_it_is_not_a_capability(self):
        assert provider().capabilities.reasoning is False


class Chunk:
    """A streamed chunk, shaped like the wire."""

    def __init__(self, model="gpt-4o", choices=(), usage=None):
        self.model = model
        self.choices = list(choices)
        self.usage = usage


class Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class StreamChoice:
    def __init__(self, delta=None, finish_reason=None):
        self.delta = delta or Delta()
        self.finish_reason = finish_reason


class CallDelta:
    def __init__(self, index, id=None, name=None, arguments=None):  # noqa: A002
        self.index = index
        self.id = id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class TestStreaming:
    async def test_usage_is_requested_or_a_streamed_agent_cannot_budget(self):
        llm = provider([Chunk(choices=[StreamChoice(Delta("hi"), "stop")])])

        [_ async for _ in llm.stream(LLMRequest(messages=(Message.user("hi"),)))]

        assert sent(llm)["stream_options"] == {"include_usage": True}

    async def test_deltas_and_a_final_assembled_response(self):
        llm = provider(
            [
                Chunk(choices=[StreamChoice(Delta("Par"))]),
                Chunk(choices=[StreamChoice(Delta("is."), "stop")]),
                # The usage-only chunk that OpenAI sends last, with no choices.
                Chunk(
                    choices=[],
                    usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                ),
            ]
        )

        events = [e async for e in llm.stream(LLMRequest(messages=(Message.user("hi"),)))]

        assert [e.text for e in events if isinstance(e, TextDelta)] == ["Par", "is."]
        response = end_of(events)
        assert response.text == "Paris."
        assert response.usage.output_tokens == 5

    async def test_parallel_tool_calls_are_assembled_by_index(self):
        llm = provider(
            [
                Chunk(
                    choices=[
                        StreamChoice(
                            Delta(
                                tool_calls=[
                                    CallDelta(0, "c1", "search", '{"q": '),
                                    CallDelta(1, "c2", "fetch", '{"url": '),
                                ]
                            )
                        )
                    ]
                ),
                Chunk(
                    choices=[
                        StreamChoice(
                            Delta(
                                tool_calls=[
                                    CallDelta(0, arguments='"paris"}'),
                                    CallDelta(1, arguments='"x.com"}'),
                                ]
                            ),
                            "tool_calls",
                        )
                    ]
                ),
            ]
        )

        events = [
            e async for e in llm.stream(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))
        ]
        response = end_of(events)

        assert response.finish_reason is FinishReason.TOOL_CALLS
        assert response.tool_calls[0].arguments == {"q": "paris"}
        assert response.tool_calls[1].arguments == {"url": "x.com"}


def http_response(status: int, headers: dict[str, str] | None = None):
    return httpx.Response(
        status,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )


class TestErrorTranslation:
    async def test_rate_limit(self):
        llm = provider(
            openai.RateLimitError(
                "slow down", response=http_response(429, {"retry-after": "12"}), body=None
            )
        )

        with pytest.raises(RateLimitError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert caught.value.retry_after == 12.0
        assert caught.value.retryable

    async def test_authentication(self):
        llm = provider(
            openai.AuthenticationError("bad key", response=http_response(401), body=None)
        )

        with pytest.raises(AuthenticationError):
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

    async def test_a_connection_failure_to_a_compatible_server_names_the_base_url(self):
        # The single most common failure when pointing this adapter at a local
        # server: it is not running. The message should say where we looked.
        llm = provider(
            openai.APIConnectionError(request=http_response(0).request),
            base_url="http://localhost:8000/v1",
        )

        with pytest.raises(LLMError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert caught.value.context["base_url"] == "http://localhost:8000/v1"
        assert caught.value.retryable


def test_it_satisfies_the_port_by_shape():
    assert isinstance(provider(), LLMProvider)
