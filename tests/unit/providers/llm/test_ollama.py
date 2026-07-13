"""Tests for the Ollama adapter.

These run through a real ``httpx.MockTransport``, so the request that gets asserted
on is the one that would actually go out on the wire — headers, JSON body, NDJSON
stream parsing and all. The adapter has no SDK to hide behind, so nothing here is
stubbed above the HTTP layer.
"""

from __future__ import annotations

import json

import pytest

httpx = pytest.importorskip("httpx")

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
from sarb.errors import InvalidInputError, LLMError, RateLimitError
from sarb.providers.llm.ollama import OllamaOptions, OllamaProvider

from .assertions import end_of

SEARCH = ToolSpec(
    name="search",
    description="Search the web.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)


class Wire:
    """Captures the request Ollama would actually have received."""

    def __init__(self) -> None:
        self.body: dict[str, object] = {}

    def json(self, key: str):
        return self.body[key]


def provider(handler, **options) -> tuple[OllamaProvider, Wire]:
    wire = Wire()

    def capture(request):
        wire.body = json.loads(request.content)
        return handler(request)

    client = httpx.AsyncClient(
        base_url="http://localhost:11434", transport=httpx.MockTransport(capture)
    )
    return OllamaProvider(client, OllamaOptions(model="llama3.1:8b", **options)), wire


def replies(**body):
    """A complete (non-streamed) Ollama response."""
    payload = {
        "model": "llama3.1:8b",
        "message": {"role": "assistant", "content": "Paris."},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
        **body,
    }
    return lambda _request: httpx.Response(200, json=payload)


def streams(*lines):
    """An NDJSON stream, exactly as Ollama emits it."""
    content = "\n".join(json.dumps(line) for line in lines)
    return lambda _request: httpx.Response(200, content=content)


def fails(status: int, body=None, headers=None):
    return lambda _request: httpx.Response(
        status, json=body or {"error": "boom"}, headers=headers or {}
    )


class TestRequestTranslation:
    async def test_the_system_prompt_becomes_the_first_message(self):
        llm, wire = provider(replies())

        await llm.generate(LLMRequest(messages=(Message.user("hi"),), system="Be terse."))

        assert wire.json("messages")[0] == {"role": "system", "content": "Be terse."}

    async def test_sampling_options_go_in_the_options_block(self):
        llm, wire = provider(replies())

        await llm.generate(
            LLMRequest(
                messages=(Message.user("hi"),), temperature=0.2, max_tokens=256, seed=7, stop=("X",)
            )
        )

        assert wire.json("options") == {
            "temperature": 0.2,
            "num_predict": 256,
            "seed": 7,
            "stop": ["X"],
        }

    async def test_extra_options_pass_straight_through(self):
        # The escape hatch for knobs the framework has no neutral concept for. The
        # day it grows a `mirostat` field is the day it has an opinion about one
        # vendor's sampler.
        llm, wire = provider(replies(), extra_options={"num_ctx": 8192})

        await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert wire.json("options")["num_ctx"] == 8192

    async def test_tool_results_are_matched_by_name_because_ollama_has_no_ids(self):
        llm, wire = provider(replies())

        await llm.generate(
            LLMRequest(
                messages=(
                    Message.user("search"),
                    Message.assistant(tool_calls=(ToolCall(id="ours-1", name="search"),)),
                    Message.tool([ToolResult(call_id="ours-1", name="search", content="Paris")]),
                ),
                tools=(SEARCH,),
            )
        )

        # Our id is ours. Ollama pairs results to calls by tool name, and never sees
        # the id at all.
        assert wire.json("messages")[-1] == {
            "role": "tool",
            "tool_name": "search",
            "content": "Paris",
        }

    async def test_a_tool_choice_it_cannot_honour_raises(self):
        # Ollama has no tool_choice. A `required` that silently does not require
        # anything leaves an agent loop waiting for a call that is never coming.
        llm, _ = provider(replies())

        with pytest.raises(InvalidInputError, match="tool_choice"):
            await llm.generate(
                LLMRequest(
                    messages=(Message.user("hi"),),
                    tools=(SEARCH,),
                    tool_choice=ToolChoice.required(),
                )
            )

    async def test_a_json_schema_goes_in_the_format_field(self):
        llm, wire = provider(replies())
        schema = {"type": "object", "properties": {"city": {"type": "string"}}}

        await llm.generate(
            LLMRequest(
                messages=(Message.user("hi"),),
                response_format=ResponseFormat(kind="json_schema", schema_=schema),
            )
        )

        assert wire.json("format") == schema

    async def test_images_are_base64_bytes(self):
        llm, wire = provider(replies())

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

        assert wire.json("messages")[0]["images"] == ["iVBORw=="]

    async def test_think_and_keep_alive(self):
        llm, wire = provider(replies(), think=True, keep_alive="30m")

        await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert wire.json("think") is True
        assert wire.json("keep_alive") == "30m"


class TestResponseTranslation:
    async def test_text_and_usage(self):
        llm, _ = provider(replies())

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert response.text == "Paris."
        assert response.finish_reason is FinishReason.STOP
        assert (response.usage.input_tokens, response.usage.output_tokens) == (10, 5)

    async def test_tool_calls_arrive_whole_and_get_an_id_minted(self):
        llm, _ = provider(
            replies(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "search", "arguments": {"q": "paris"}}}],
                }
            )
        )

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))

        call = response.tool_calls[0]
        assert call.name == "search"
        assert call.arguments == {"q": "paris"}
        assert call.id  # minted by us, because Ollama sends none

    async def test_a_clean_stop_with_tool_calls_is_still_tool_calls(self):
        # Ollama says done_reason="stop" even when it is waiting for tools. An agent
        # loop reading STOP would end the turn with the tools uncalled.
        llm, _ = provider(
            replies(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "search", "arguments": {}}}],
                }
            )
        )

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))

        assert response.finish_reason is FinishReason.TOOL_CALLS

    async def test_thinking_is_reasoning_not_answer_text(self):
        llm, _ = provider(
            replies(message={"role": "assistant", "content": "Paris.", "thinking": "hmm"}),
            think=True,
        )

        response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert response.text == "Paris."
        assert response.message.reasoning == "hmm"


class TestStreaming:
    async def test_ndjson_lines_become_events(self):
        llm, _ = provider(
            streams(
                {"model": "llama3.1:8b", "message": {"content": "Par"}, "done": False},
                {"model": "llama3.1:8b", "message": {"content": "is."}, "done": False},
                {
                    "model": "llama3.1:8b",
                    "message": {"content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            )
        )

        events = [e async for e in llm.stream(LLMRequest(messages=(Message.user("hi"),)))]

        assert [e.text for e in events if isinstance(e, TextDelta)] == ["Par", "is."]
        response = end_of(events)
        assert response.text == "Paris."
        assert response.usage.output_tokens == 5

    async def test_a_streamed_tool_call(self):
        llm, _ = provider(
            streams(
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "search", "arguments": {"q": "paris"}}}
                        ],
                    },
                    "done": False,
                },
                {"message": {"content": ""}, "done": True, "done_reason": "stop"},
            )
        )

        events = [
            e async for e in llm.stream(LLMRequest(messages=(Message.user("hi"),), tools=(SEARCH,)))
        ]
        response = end_of(events)

        assert response.finish_reason is FinishReason.TOOL_CALLS
        assert response.tool_calls[0].arguments == {"q": "paris"}


class TestErrorTranslation:
    async def test_ollamas_own_error_text_survives(self):
        # A 404 from Ollama nearly always means the model has not been pulled — and
        # it says so, but only in the body.
        llm, _ = provider(fails(404, {"error": 'model "llama3.1:8b" not found, try pulling it'}))

        with pytest.raises(LLMError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert "try pulling it" in str(caught.value)
        assert not caught.value.retryable

    async def test_a_server_error_is_retryable(self):
        llm, _ = provider(fails(503))

        with pytest.raises(LLMError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert caught.value.retryable

    async def test_rate_limiting_behind_a_proxy(self):
        llm, _ = provider(fails(429, headers={"retry-after": "5"}))

        with pytest.raises(RateLimitError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert caught.value.retry_after == 5.0

    async def test_a_daemon_that_is_not_running_says_so(self):
        def refuse(request):
            raise httpx.ConnectError("connection refused", request=request)

        llm, _ = provider(refuse)

        with pytest.raises(LLMError) as caught:
            await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

        assert "Is it running?" in str(caught.value)
        assert caught.value.retryable


def test_it_satisfies_the_port_by_shape():
    llm, _ = provider(replies())
    assert isinstance(llm, LLMProvider)
