"""Tests for the LLM request/response models and the streaming event union."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from sarb.contracts import (
    LLMRequest,
    LLMResponse,
    Message,
    ResponseFormat,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolChoice,
    ToolSpec,
    Usage,
)

SEARCH = ToolSpec(
    name="search",
    description="Search the web.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)


class TestLLMRequest:
    def test_minimal_request(self):
        request = LLMRequest(messages=(Message.user("hi"),))

        assert request.tool_choice.mode == "auto"
        # None, not 0.0: the framework must not silently pick a value that changes
        # how a model behaves.
        assert request.temperature is None

    def test_a_request_needs_a_message(self):
        with pytest.raises(ValidationError, match="at least one message"):
            LLMRequest(messages=())

    def test_tool_choice_needs_tools(self):
        with pytest.raises(ValidationError, match="needs tools"):
            LLMRequest(messages=(Message.user("hi"),), tool_choice=ToolChoice.required())

    def test_named_tool_choice_must_name_a_supplied_tool(self):
        with pytest.raises(ValidationError, match="not in tools"):
            LLMRequest(
                messages=(Message.user("hi"),),
                tools=(SEARCH,),
                tool_choice=ToolChoice.tool("fetch"),
            )

    def test_named_tool_choice_accepts_a_supplied_tool(self):
        request = LLMRequest(
            messages=(Message.user("hi"),),
            tools=(SEARCH,),
            tool_choice=ToolChoice.tool("search"),
        )

        assert request.tool_choice.name == "search"

    def test_evolve_makes_retrying_with_different_params_cheap(self):
        request = LLMRequest(messages=(Message.user("hi"),), temperature=0.7)

        assert request.evolve(temperature=0.0).temperature == 0.0
        assert request.temperature == 0.7  # original untouched

    def test_temperature_is_bounded(self):
        with pytest.raises(ValidationError):
            LLMRequest(messages=(Message.user("hi"),), temperature=5.0)


class TestToolChoice:
    def test_a_name_requires_tool_mode(self):
        with pytest.raises(ValidationError, match="mode"):
            ToolChoice(mode="auto", name="search")

    def test_tool_mode_requires_a_name(self):
        with pytest.raises(ValidationError, match="mode"):
            ToolChoice(mode="tool")


class TestResponseFormat:
    def test_json_schema_needs_a_schema(self):
        # Otherwise it is a silent no-op: the caller thinks output is constrained
        # and it is not.
        with pytest.raises(ValidationError, match="schema"):
            ResponseFormat(kind="json_schema")

    def test_a_schema_needs_json_schema_kind(self):
        with pytest.raises(ValidationError, match="schema"):
            ResponseFormat(kind="text", schema_={"type": "object"})

    def test_schema_serialises_under_its_json_name(self):
        # `schema` collides with a Pydantic method, so the attribute carries a
        # trailing underscore. Config and JSON still say `schema`.
        fmt = ResponseFormat(kind="json_schema", schema_={"type": "object"})

        assert fmt.model_dump(by_alias=True)["schema"] == {"type": "object"}
        assert (
            ResponseFormat.model_validate({"kind": "json_schema", "schema": {"type": "object"}})
            == fmt
        )


class TestLLMResponse:
    def test_convenience_accessors(self):
        response = LLMResponse(
            message=Message.assistant("Paris.", tool_calls=(ToolCall(id="c1", name="search"),)),
            model="claude-sonnet-5",
            usage=Usage(input_tokens=10, output_tokens=2),
        )

        assert response.text == "Paris."
        assert response.tool_calls[0].name == "search"
        assert response.usage.total_tokens == 12


class TestStreamEvents:
    def test_events_are_discriminated_and_round_trip(self):
        adapter: TypeAdapter[StreamEvent] = TypeAdapter(StreamEvent)

        event = adapter.validate_python({"kind": "text", "text": "Par"})

        assert isinstance(event, TextDelta)
        assert adapter.dump_python(event)["kind"] == "text"

    def test_stream_end_carries_the_assembled_response(self):
        # This is what stops every consumer from reimplementing delta assembly,
        # each one slightly differently.
        end = StreamEnd(response=LLMResponse(message=Message.assistant("Paris."), model="m"))

        assert end.response.text == "Paris."

    def test_an_unknown_event_kind_is_rejected(self):
        with pytest.raises(ValidationError):
            TypeAdapter(StreamEvent).validate_python({"kind": "telepathy"})
