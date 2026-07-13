"""Tests for the Anthropic adapter.

Fixtures are built from **Anthropic's own Pydantic models**, not from hand-written
dicts. That distinction is the whole point of this file: a hand-written fixture can
be wrong in exactly the same way the adapter is wrong, and the pair of them will
agree with each other happily, forever, while production fails. Constructing an
``anthropic.types.Message`` makes the SDK validate the shape we claim to receive.

No network, no key: the client is injected, which is why the adapter takes one.
"""

from __future__ import annotations

import pytest

anthropic = pytest.importorskip("anthropic")

import httpx
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock, Usage

from sarb.contracts import (
    FinishReason,
    ImagePart,
    LLMProvider,
    LLMRequest,
    Message,
    ReasoningPart,
    ResponseFormat,
    Role,
    StreamStart,
    TextDelta,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolSpec,
)
from sarb.errors import (
    AuthenticationError,
    InvalidInputError,
    LLMError,
    ProviderTimeoutError,
    RateLimitError,
)
from sarb.providers.llm.anthropic import AnthropicOptions, AnthropicProvider

from .assertions import end_of, reasoning_of

SEARCH = ToolSpec(
    name="search",
    description="Search the web.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)


# --------------------------------------------------------------------------- #
# A recording stand-in for anthropic.AsyncAnthropic
# --------------------------------------------------------------------------- #


class FakeMessages:
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
        self.messages = FakeMessages(result)
        self.closed = False

    async def close(self):
        self.closed = True


async def _aiter(events):
    for event in events:
        yield event


def provider(result=None, **options) -> AnthropicProvider:
    return AnthropicProvider(
        FakeClient(result),  # type: ignore[arg-type]
        AnthropicOptions(model="claude-opus-4-8", **options),
    )


def sent(llm):
    """The kwargs the adapter would actually have sent to Anthropic."""
    return llm._client.messages.params


def reply(*content, stop_reason="end_turn", **usage) -> AnthropicMessage:
    """A complete Anthropic response, validated by Anthropic's own model."""
    return AnthropicMessage(
        id="msg_1",
        model="claude-opus-4-8",
        role="assistant",
        type="message",
        stop_reason=stop_reason,
        content=list(content),
        usage=Usage(input_tokens=usage.get("input", 10), output_tokens=usage.get("output", 5)),
    )


def text(value: str) -> TextBlock:
    return TextBlock(type="text", text=value, citations=None)


# --------------------------------------------------------------------------- #
# Outbound: framework -> Anthropic
# --------------------------------------------------------------------------- #


class TestRequestTranslation:
    async def test_the_system_prompt_becomes_a_parameter(self):
        # Not a message. Anthropic has no system role, which is the shape the port
        # was designed around.
        llm = provider(reply(text("hi")))

        await llm.generate(LLMRequest(messages=(Message.user("hi"),), system="Be terse."))

        params = sent(llm)
        assert params["system"] == "Be terse."
        assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    async def test_max_tokens_is_always_sent(self):
        # Anthropic rejects a request without it, so the adapter cannot simply omit
        # what the caller did not set.
        llm = provider(reply(text("hi")), max_tokens=2048)

        await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert sent(llm)["max_tokens"] == 2048

    async def test_a_request_max_tokens_wins_over_the_configured_default(self):
        llm = provider(reply(text("hi")), max_tokens=2048)

        await llm.generate(LLMRequest(messages=(Message.user("hi"),), max_tokens=100))

        assert sent(llm)["max_tokens"] == 100

    async def test_a_tool_turn_becomes_tool_result_blocks_in_one_user_message(self):
        # The framework models one tool turn holding every result. Anthropic agrees,
        # so this is the adapter that has the easy side of that argument.
        llm = provider(reply(text("done")))

        await llm.generate(
            LLMRequest(
                messages=(
                    Message.user("search"),
                    Message.assistant(tool_calls=(ToolCall(id="tu_1", name="search"),)),
                    Message.tool(
                        [
                            ToolResult(call_id="tu_1", name="search", content="Paris"),
                            ToolResult(call_id="tu_2", name="search", content="no", is_error=True),
                        ]
                    ),
                ),
                tools=(SEARCH,),
            )
        )

        tool_turn = sent(llm)["messages"][-1]
        assert tool_turn["role"] == "user"
        assert len(tool_turn["content"]) == 2
        assert tool_turn["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": "Paris",
            "is_error": False,
        }
        assert tool_turn["content"][1]["is_error"] is True

    async def test_an_assistant_tool_call_round_trips_as_a_tool_use_block(self):
        llm = provider(reply(text("done")))

        await llm.generate(
            LLMRequest(
                messages=(
                    Message.user("search"),
                    Message.assistant(
                        "Looking.",
                        tool_calls=(ToolCall(id="tu_1", name="search", arguments={"q": "paris"}),),
                    ),
                ),
                tools=(SEARCH,),
            )
        )

        blocks = sent(llm)["messages"][1]["content"]
        assert blocks[0] == {"type": "text", "text": "Looking."}
        assert blocks[1] == {
            "type": "tool_use",
            "id": "tu_1",
            "name": "search",
            "input": {"q": "paris"},
        }

    async def test_signed_reasoning_is_echoed_back_verbatim(self):
        # Anthropic validates the signature and rejects the turn if it is altered.
        llm = provider(reply(text("done")), thinking=True)

        await llm.generate(
            LLMRequest(
                messages=(
                    Message.user("hi"),
                    Message(
                        role=Role.ASSISTANT,
                        content=(ReasoningPart(text="hmm", signature="sig-abc"),),
                    ),
                )
            )
        )

        assert sent(llm)["messages"][1]["content"][0] == {
            "type": "thinking",
            "thinking": "hmm",
            "signature": "sig-abc",
        }

    async def test_unsigned_reasoning_is_dropped_rather_than_rejected(self):
        # Reasoning that came from another provider has no Anthropic signature.
        # A framework makes exactly that easy, so sending it would break the call.
        llm = provider(reply(text("done")))

        await llm.generate(
            LLMRequest(
                messages=(
                    Message.user("hi"),
                    Message(role=Role.ASSISTANT, content=(ReasoningPart(text="from gemini"),)),
                )
            )
        )

        assert sent(llm)["messages"][1]["content"] == []

    async def test_an_inline_image_is_base64_encoded(self):
        llm = provider(reply(text("a cat")))

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

        source = sent(llm)["messages"][0]["content"][0]["source"]
        assert source == {"type": "base64", "media_type": "image/png", "data": "iVBORw=="}

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            (ToolChoice.auto(), {"type": "auto"}),
            (ToolChoice.none(), {"type": "none"}),
            (ToolChoice.required(), {"type": "any"}),  # Anthropic calls it "any"
            (ToolChoice.tool("search"), {"type": "tool", "name": "search"}),
        ],
    )
    async def test_tool_choice(self, choice, expected):
        llm = provider(reply(text("hi")))

        await llm.generate(
            LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,), tool_choice=choice)
        )

        assert sent(llm)["tool_choice"] == expected

    async def test_thinking_and_effort_reach_the_api(self):
        llm = provider(reply(text("hi")), thinking=True, effort="high")

        await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        params = sent(llm)
        assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert params["output_config"]["effort"] == "high"

    async def test_a_json_schema_becomes_an_output_config_format(self):
        llm = provider(reply(text("{}")))
        schema = {"type": "object", "properties": {"city": {"type": "string"}}}

        await llm.generate(
            LLMRequest(
                messages=(Message.user("hi"),),
                response_format=ResponseFormat(kind="json_schema", schema_=schema),
            )
        )

        assert sent(llm)["output_config"]["format"] == {
            "type": "json_schema",
            "schema": schema,
        }

    async def test_a_format_it_cannot_express_raises_rather_than_being_dropped(self):
        # Silently dropping this leaves the caller believing output is constrained
        # when it is not — and they find out later, in a JSON parse error, a long way
        # from the cause.
        llm = provider(reply(text("hi")))

        with pytest.raises(InvalidInputError, match="json_schema"):
            await llm.generate(
                LLMRequest(
                    messages=(Message.user("hi"),), response_format=ResponseFormat(kind="json")
                )
            )


# --------------------------------------------------------------------------- #
# Inbound: Anthropic -> framework
# --------------------------------------------------------------------------- #


class TestResponseTranslation:
    async def test_text(self):
        llm = provider(reply(text("Paris.")))

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert response.text == "Paris."
        assert response.finish_reason is FinishReason.STOP
        assert response.model == "claude-opus-4-8"

    async def test_tool_use_blocks_become_tool_calls(self):
        llm = provider(
            reply(
                text("Looking."),
                ToolUseBlock(type="tool_use", id="tu_1", name="search", input={"q": "paris"}),
                stop_reason="tool_use",
            )
        )

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))

        assert response.finish_reason is FinishReason.TOOL_CALLS
        assert response.text == "Looking."
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].arguments == {"q": "paris"}

    async def test_thinking_becomes_reasoning_and_stays_out_of_the_answer(self):
        llm = provider(
            reply(ThinkingBlock(type="thinking", thinking="hmm", signature="sig"), text("Paris.")),
            thinking=True,
        )

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert response.text == "Paris."
        assert response.message.reasoning == "hmm"
        assert reasoning_of(response).signature == "sig"

    async def test_the_cache_tokens_are_added_into_the_input_total(self):
        # The whole reason `Usage` pins its conventions. Anthropic's `input_tokens`
        # counts only what came *after* the last cache breakpoint — the documented total
        # is `input + cache_read + cache_creation`. OpenAI's `prompt_tokens`, meanwhile,
        # already includes its cached tokens.
        #
        # Copy each vendor's number straight through and `Usage.input_tokens` means two
        # different things depending on who served the call, which is a cost figure that
        # is silently wrong for one of them. So the adapter adds them up, and this is the
        # test that says so.
        message = reply(text("hi"))  # 10 fresh input, 5 output
        message.usage.cache_read_input_tokens = 7
        message.usage.cache_creation_input_tokens = 3
        llm = provider(message)

        usage = (await llm.generate(LLMRequest(messages=(Message.user("hi"),)))).usage

        assert usage.input_tokens == 20  # 10 fresh + 7 read + 3 written, not 10
        assert usage.cached_input_tokens == 7
        assert usage.cache_write_input_tokens == 3
        assert usage.output_tokens == 5

        # And the part that actually gets billed at the full rate is what is left.
        assert usage.uncached_input_tokens == 10

    @pytest.mark.parametrize(
        ("stop_reason", "expected"),
        [
            ("end_turn", FinishReason.STOP),
            ("stop_sequence", FinishReason.STOP),
            ("max_tokens", FinishReason.LENGTH),
            ("tool_use", FinishReason.TOOL_CALLS),
            ("refusal", FinishReason.CONTENT_FILTER),
        ],
    )
    async def test_stop_reasons(self, stop_reason, expected):
        llm = provider(reply(text("hi"), stop_reason=stop_reason))

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert response.finish_reason is expected


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


class Raw:
    """A raw stream event, as a plain object with the wire's field names."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class TestStreaming:
    async def test_text_deltas_and_an_assembled_end(self):
        llm = provider(
            [
                Raw(type="message_start", message=reply(text(""), **{"input": 10})),
                Raw(type="content_block_start", index=0, content_block=Raw(type="text")),
                Raw(type="content_block_delta", index=0, delta=Raw(type="text_delta", text="Par")),
                Raw(type="content_block_delta", index=0, delta=Raw(type="text_delta", text="is.")),
                Raw(
                    type="message_delta",
                    delta=Raw(stop_reason="end_turn"),
                    usage=Raw(output_tokens=42),
                ),
                Raw(type="message_stop"),
            ]
        )

        events = [e async for e in llm.stream(LLMRequest(messages=(Message.user("hi"),)))]

        assert isinstance(events[0], StreamStart)
        assert [e.text for e in events if isinstance(e, TextDelta)] == ["Par", "is."]

        response = end_of(events)
        # The whole point of StreamEnd: nobody downstream reassembles this.
        assert response.text == "Paris."
        assert response.finish_reason is FinishReason.STOP
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 42

    async def test_a_streamed_tool_call_is_assembled_from_partial_json(self):
        llm = provider(
            [
                Raw(type="message_start", message=reply(text(""))),
                Raw(
                    type="content_block_start",
                    index=0,
                    content_block=Raw(type="tool_use", id="tu_1", name="search"),
                ),
                Raw(
                    type="content_block_delta",
                    index=0,
                    delta=Raw(type="input_json_delta", partial_json='{"q": '),
                ),
                Raw(
                    type="content_block_delta",
                    index=0,
                    delta=Raw(type="input_json_delta", partial_json='"paris"}'),
                ),
                Raw(
                    type="message_delta",
                    delta=Raw(stop_reason="tool_use"),
                    usage=Raw(output_tokens=9),
                ),
            ]
        )

        events = [
            e async for e in llm.stream(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))
        ]
        response = end_of(events)

        assert response.finish_reason is FinishReason.TOOL_CALLS
        assert response.tool_calls[0].arguments == {"q": "paris"}

    async def test_a_signature_reaches_the_assembled_response_without_its_own_event(self):
        # It is a token to echo back, not content anyone renders — but losing it
        # breaks the *next* turn, which is a long way from here.
        llm = provider(
            [
                Raw(type="message_start", message=reply(text(""))),
                Raw(
                    type="content_block_delta",
                    index=0,
                    delta=Raw(type="thinking_delta", thinking="hmm"),
                ),
                Raw(
                    type="content_block_delta",
                    index=0,
                    delta=Raw(type="signature_delta", signature="sig-abc"),
                ),
                Raw(
                    type="message_delta",
                    delta=Raw(stop_reason="end_turn"),
                    usage=Raw(output_tokens=1),
                ),
            ],
            thinking=True,
        )

        events = [e async for e in llm.stream(LLMRequest(messages=(Message.user("hi"),)))]
        response = end_of(events)

        assert reasoning_of(response).signature == "sig-abc"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def http_response(status: int, headers: dict[str, str] | None = None):
    """A real ``httpx.Response`` — the vendor's exceptions will not accept less."""
    return httpx.Response(
        status,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


class TestErrorTranslation:
    """An engine must never have to ``import anthropic`` to catch a rate limit."""

    async def test_rate_limit(self):
        vendor = anthropic.RateLimitError(
            "slow down",
            response=http_response(429, {"retry-after": "30"}),
            body=None,
        )
        llm = provider(vendor)

        with pytest.raises(RateLimitError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert caught.value.retryable
        assert caught.value.retry_after == 30.0
        assert caught.value.context["model"] == "claude-opus-4-8"
        assert caught.value.__cause__ is vendor

    async def test_authentication(self):
        llm = provider(
            anthropic.AuthenticationError("bad key", response=http_response(401), body=None)
        )

        with pytest.raises(AuthenticationError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert not caught.value.retryable

    async def test_a_server_error_is_retryable(self):
        llm = provider(
            anthropic.InternalServerError("boom", response=http_response(529), body=None)
        )

        with pytest.raises(LLMError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert caught.value.retryable

    async def test_timeout(self):
        llm = provider(anthropic.APITimeoutError(request=None))

        with pytest.raises(ProviderTimeoutError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert caught.value.retryable

    async def test_an_unrecognised_failure_is_still_an_llm_error(self):
        llm = provider(RuntimeError("something else entirely"))

        with pytest.raises(LLMError):
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))


class TestPort:
    def test_it_satisfies_the_port_by_shape(self):
        assert isinstance(provider(), LLMProvider)

    def test_capabilities_describe_the_api_not_a_guess_at_the_model(self):
        caps = provider(thinking=True, max_input_tokens=200_000).capabilities

        assert caps.tools
        assert caps.vision
        assert caps.json_schema
        assert caps.reasoning
        assert caps.max_input_tokens == 200_000

    async def test_closing_closes_the_client(self):
        llm = provider()

        await llm.aclose()

        assert llm._client.closed  # type: ignore[attr-defined]
