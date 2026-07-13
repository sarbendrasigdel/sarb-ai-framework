"""Tests for the machinery every LLM adapter shares.

This file matters more than it looks. Four adapters delegate their trickiest
behaviour here — argument parsing, delta assembly — so a bug in this module is a bug
in every provider at once, including the ones nobody has written yet.

Error classification is not specific to language models and moved up a level, to
``tests/unit/providers/test_shared.py``.
"""

from __future__ import annotations

import pytest

from sarb.contracts import FinishReason, ReasoningPart, Usage
from sarb.providers.llm._shared import LLMOptions, ResponseBuilder, parse_arguments


class TestParseArguments:
    def test_valid_json_object(self):
        arguments, malformed = parse_arguments('{"city": "Paris"}')

        assert arguments == {"city": "Paris"}
        assert malformed is None

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_absent_arguments_are_not_malformed(self, raw):
        # A tool called with no arguments is a normal thing to do. Reporting it as
        # malformed would send the model an error it cannot act on.
        assert parse_arguments(raw) == ({}, None)

    def test_broken_json_is_carried_not_raised(self):
        arguments, malformed = parse_arguments('{"city": "Par')

        assert arguments == {}
        assert malformed == '{"city": "Par'

    @pytest.mark.parametrize("raw", ["[1, 2]", '"Paris"', "null", "42"])
    def test_valid_json_that_is_not_an_object_is_malformed(self, raw):
        # Tool arguments are named. A bare array parses as JSON and is still not
        # something you can call a function with.
        _, malformed = parse_arguments(raw)

        assert malformed == raw


class TestResponseBuilder:
    def test_text_deltas_concatenate(self):
        builder = ResponseBuilder("m")
        builder.add_text("Par")
        builder.add_text("is.")

        assert builder.build().text == "Paris."

    def test_reasoning_is_not_answer_text(self):
        # If these two ever merge, a UI streams the model's scratch work to the user
        # as if it were the answer.
        builder = ResponseBuilder("m")
        builder.add_reasoning("Let me think...")
        builder.add_text("Paris.")

        response = builder.build()

        assert response.text == "Paris."
        assert response.message.reasoning == "Let me think..."

    def test_a_signature_arrives_after_the_reasoning_it_signs(self):
        # Anthropic sends signature_delta as a separate event, once the thinking
        # block is done. Losing it means the next turn of the conversation is
        # rejected by the API.
        builder = ResponseBuilder("m")
        builder.add_reasoning("thinking")
        builder.set_signature("sig-abc")

        part = builder.build().message.content[0]
        assert isinstance(part, ReasoningPart)
        assert part.signature == "sig-abc"

    def test_parallel_tool_call_fragments_do_not_bleed_into_each_other(self):
        # The bug this exists to prevent: two calls streamed interleaved, their
        # argument fragments concatenated into one unparseable string.
        builder = ResponseBuilder("m")
        builder.add_tool_call_delta(0, call_id="a", name="search")
        builder.add_tool_call_delta(1, call_id="b", name="fetch")
        builder.add_tool_call_delta(0, arguments_delta='{"q": ')
        builder.add_tool_call_delta(1, arguments_delta='{"url": ')
        builder.add_tool_call_delta(0, arguments_delta='"paris"}')
        builder.add_tool_call_delta(1, arguments_delta='"x.com"}')

        calls = builder.build().tool_calls

        assert [call.name for call in calls] == ["search", "fetch"]
        assert calls[0].arguments == {"q": "paris"}
        assert calls[1].arguments == {"url": "x.com"}

    def test_a_call_with_broken_arguments_survives_assembly(self):
        builder = ResponseBuilder("m")
        builder.add_tool_call_delta(0, call_id="a", name="search", arguments_delta='{"q":')

        call = builder.build().tool_calls[0]

        assert call.is_malformed
        assert call.malformed_arguments == '{"q":'
        assert call.arguments == {}

    def test_a_call_with_no_id_still_gets_one(self):
        # Ollama issues no ids at all. An empty id would silently pair every
        # parallel call with every result.
        builder = ResponseBuilder("m")
        builder.add_tool_call_delta(0, name="search")

        assert builder.build().tool_calls[0].id

    def test_complete_calls_skip_json_parsing(self):
        builder = ResponseBuilder("m")
        builder.add_tool_call(call_id="a", name="search", arguments={"q": "paris"})

        call = builder.build().tool_calls[0]

        assert call.arguments == {"q": "paris"}
        assert not call.is_malformed

    def test_build_records_what_the_provider_actually_served(self):
        builder = ResponseBuilder("claude-sonnet-5")

        response = builder.build(
            model="claude-sonnet-5-20260101",
            usage=Usage(input_tokens=10, output_tokens=2),
            finish_reason=FinishReason.TOOL_CALLS,
        )

        assert response.model == "claude-sonnet-5-20260101"
        assert response.usage.total_tokens == 12
        assert response.finish_reason is FinishReason.TOOL_CALLS

    def test_an_empty_stream_still_produces_a_response(self):
        # StreamEnd promises an assembled response even when nothing was generated.
        response = ResponseBuilder("m").build()

        assert response.text == ""
        assert response.message.tool_calls == ()


class TestLLMOptions:
    def test_a_model_must_be_named(self):
        # No default. A framework that picks a model for you picks the wrong one and
        # bills you for it.
        # Options come from YAML at runtime, so this must be a *runtime* failure —
        # the type checker cannot see a config file.
        with pytest.raises(ValueError, match="model"):
            LLMOptions()  # type: ignore[call-arg]

    def test_unknown_options_are_rejected(self):
        # A typo in YAML must not be silently ignored — `temprature: 0.7` would leave
        # the operator convinced they had set something.
        with pytest.raises(ValueError, match="temprature"):
            LLMOptions(model="m", temprature=0.7)  # type: ignore[call-arg]
