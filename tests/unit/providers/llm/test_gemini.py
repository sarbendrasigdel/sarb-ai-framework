"""Tests for the Gemini adapter.

Gemini's shape is the furthest from the framework's, so this is where translation
bugs are most likely and where the fixtures being *Google's own types* matters most.
Three of its quirks get a test each, because each one is a silent failure rather
than a loud one: the assistant is called ``model``, reasoning is a boolean flag on
an otherwise ordinary text part, and a turn that asked for tools still reports a
clean stop.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.genai")

from google.genai import errors, types

from sarb.contracts import (
    FinishReason,
    LLMProvider,
    LLMRequest,
    Message,
    ResponseFormat,
    TextDelta,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolSpec,
)
from sarb.errors import AuthenticationError, LLMError, RateLimitError
from sarb.providers.llm.gemini import GeminiOptions, GeminiProvider

from .assertions import end_of

SEARCH = ToolSpec(
    name="search",
    description="Search the web.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)


class FakeModels:
    def __init__(self, result):
        self._result = result
        self.contents = None
        self.config = None

    async def generate_content(self, *, model, contents, config):
        self.contents, self.config = contents, config
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result

    async def generate_content_stream(self, *, model, contents, config):
        self.contents, self.config = contents, config
        if isinstance(self._result, BaseException):
            raise self._result
        return _aiter(self._result)


async def _aiter(chunks):
    for chunk in chunks:
        yield chunk


class FakeClient:
    def __init__(self, result):
        self.aio = type("Aio", (), {"models": FakeModels(result)})()


def provider(result=None, **options) -> GeminiProvider:
    return GeminiProvider(
        FakeClient(result),  # type: ignore[arg-type]
        GeminiOptions(model="gemini-2.0-flash", **options),
    )


def sent(llm):
    """The contents and config the adapter would actually have sent to Gemini."""
    return llm._client.aio.models


def reply(*parts, finish_reason="STOP") -> types.GenerateContentResponse:
    """A complete Gemini response, validated by Google's own model."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5, thoughts_token_count=2
        ),
    )


class TestRequestTranslation:
    async def test_the_assistant_is_renamed_to_model(self):
        # A one-word rename — and exactly the kind of thing that would otherwise
        # leak into an engine as `if provider == "gemini"`.
        llm = provider(reply(types.Part(text="Paris.")))

        await llm.generate(LLMRequest(messages=(Message.user("hi"), Message.assistant("Bonjour."))))

        assert [content.role for content in sent(llm).contents] == ["user", "model"]

    async def test_the_system_prompt_becomes_a_system_instruction(self):
        llm = provider(reply(types.Part(text="Paris.")))

        await llm.generate(LLMRequest(messages=(Message.user("hi"),), system="Be terse."))

        assert sent(llm).config.system_instruction == "Be terse."

    async def test_a_tool_result_becomes_a_function_response_part_in_a_user_turn(self):
        llm = provider(reply(types.Part(text="done")))

        await llm.generate(
            LLMRequest(
                messages=(
                    Message.user("search"),
                    Message.assistant(tool_calls=(ToolCall(id="c1", name="search"),)),
                    Message.tool([ToolResult(call_id="c1", name="search", content="Paris")]),
                ),
                tools=(SEARCH,),
            )
        )

        turn = sent(llm).contents[-1]
        assert turn.role == "user"
        response = turn.parts[0].function_response
        assert response.name == "search"
        assert response.id == "c1"
        assert response.response == {"result": "Paris"}

    async def test_a_failed_tool_says_so_under_error(self):
        llm = provider(reply(types.Part(text="done")))

        await llm.generate(
            LLMRequest(
                messages=(
                    Message.user("search"),
                    Message.assistant(tool_calls=(ToolCall(id="c1", name="search"),)),
                    Message.tool(
                        [ToolResult(call_id="c1", name="search", content="boom", is_error=True)]
                    ),
                ),
                tools=(SEARCH,),
            )
        )

        assert sent(llm).contents[-1].parts[0].function_response.response == {"error": "boom"}

    async def test_tools_are_declared_with_plain_json_schema(self):
        # `parameters_json_schema`, not `parameters`. The latter takes Gemini's own
        # Schema dialect, which would mean maintaining a converter for a language
        # nobody else speaks.
        llm = provider(reply(types.Part(text="hi")))

        await llm.generate(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))

        declaration = sent(llm).config.tools[0].function_declarations[0]
        assert declaration.name == "search"
        assert declaration.parameters_json_schema == SEARCH.parameters

    async def test_automatic_function_calling_is_turned_off(self):
        # The SDK will happily execute callables it recognises. Turning it off
        # explicitly means a future SDK default cannot start calling tools behind the
        # agent loop's back.
        llm = provider(reply(types.Part(text="hi")))

        await llm.generate(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))

        assert sent(llm).config.automatic_function_calling.disable is True

    @pytest.mark.parametrize(
        ("choice", "mode", "allowed"),
        [
            (ToolChoice.auto(), "AUTO", None),
            (ToolChoice.none(), "NONE", None),
            (ToolChoice.required(), "ANY", None),
            (ToolChoice.tool("search"), "ANY", ["search"]),
        ],
    )
    async def test_tool_choice(self, choice, mode, allowed):
        llm = provider(reply(types.Part(text="hi")))

        await llm.generate(
            LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,), tool_choice=choice)
        )

        config = sent(llm).config.tool_config.function_calling_config
        assert config.mode == mode
        assert config.allowed_function_names == allowed

    async def test_a_json_schema_sets_the_mime_type_too(self):
        # Without the mime type Gemini ignores the schema entirely.
        llm = provider(reply(types.Part(text="{}")))
        schema = {"type": "object", "properties": {"city": {"type": "string"}}}

        await llm.generate(
            LLMRequest(
                messages=(Message.user("hi"),),
                response_format=ResponseFormat(kind="json_schema", schema_=schema),
            )
        )

        assert sent(llm).config.response_mime_type == "application/json"
        assert sent(llm).config.response_json_schema == schema


class TestResponseTranslation:
    async def test_text_and_usage(self):
        llm = provider(reply(types.Part(text="Paris.")))

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert response.text == "Paris."
        assert response.finish_reason is FinishReason.STOP
        assert response.usage.input_tokens == 10
        assert response.usage.reasoning_tokens == 2

    async def test_a_thought_part_is_reasoning_not_an_answer(self):
        # `thought` is a *flag* on an otherwise ordinary text part. Miss it and the
        # model's scratch work is streamed to the user as the answer.
        llm = provider(
            reply(
                types.Part(text="Let me think...", thought=True),
                types.Part(text="Paris."),
            ),
            thinking=True,
        )

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert response.text == "Paris."
        assert response.message.reasoning == "Let me think..."

    async def test_a_function_call_becomes_a_tool_call_and_corrects_the_finish_reason(self):
        # Gemini reports STOP even while it waits for tools. Left alone, an agent
        # loop ends the turn with the tools uncalled.
        llm = provider(
            reply(
                types.Part(
                    function_call=types.FunctionCall(name="search", args={"q": "paris"}, id="fc_1")
                ),
                finish_reason="STOP",
            )
        )

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))

        assert response.finish_reason is FinishReason.TOOL_CALLS
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].arguments == {"q": "paris"}

    async def test_a_call_without_an_id_still_gets_one(self):
        llm = provider(reply(types.Part(function_call=types.FunctionCall(name="search", args={}))))

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))

        assert response.tool_calls[0].id

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("STOP", FinishReason.STOP),
            ("MAX_TOKENS", FinishReason.LENGTH),
            ("SAFETY", FinishReason.CONTENT_FILTER),
            ("PROHIBITED_CONTENT", FinishReason.CONTENT_FILTER),
            ("MALFORMED_FUNCTION_CALL", FinishReason.ERROR),
        ],
    )
    async def test_finish_reasons(self, reason, expected):
        llm = provider(reply(types.Part(text="hi"), finish_reason=reason))

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert response.finish_reason is expected

    async def test_a_blocked_response_with_no_candidates_does_not_explode(self):
        llm = provider(types.GenerateContentResponse(candidates=[]))

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert response.text == ""


class TestStreaming:
    async def test_text_deltas_and_an_assembled_end(self):
        llm = provider(
            [
                reply(types.Part(text="Par"), finish_reason=None),
                reply(types.Part(text="is."), finish_reason="STOP"),
            ]
        )

        events = [e async for e in llm.stream(LLMRequest(messages=(Message.user("hi"),)))]

        assert [e.text for e in events if isinstance(e, TextDelta)] == ["Par", "is."]
        response = end_of(events)
        assert response.text == "Paris."
        assert response.usage.input_tokens == 10


class TestErrorTranslation:
    def _api_error(self, code: int, message: str):
        return errors.APIError(code, {"error": {"message": message, "status": "ERROR"}})

    async def test_rate_limit(self):
        llm = provider(self._api_error(429, "quota exceeded"))

        with pytest.raises(RateLimitError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert caught.value.retryable

    async def test_authentication(self):
        llm = provider(self._api_error(401, "bad key"))

        with pytest.raises(AuthenticationError):
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

    async def test_a_server_error_is_retryable(self):
        llm = provider(self._api_error(503, "unavailable"))

        with pytest.raises(LLMError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert caught.value.retryable


def test_it_satisfies_the_port_by_shape():
    assert isinstance(provider(), LLMProvider)
