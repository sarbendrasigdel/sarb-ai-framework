"""Tests for the conversation model.

The role invariants are the valuable ones: they are enforced here, once, so that
four adapters do not each discover a malformed message differently — and so that
one of them does not forget to check at all.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sarb.contracts import (
    FinishReason,
    ImagePart,
    Message,
    ReasoningPart,
    Role,
    TextPart,
    ToolCall,
    ToolResult,
    Usage,
)


class TestConstructors:
    def test_user(self):
        message = Message.user("What is the capital of France?")

        assert message.role is Role.USER
        assert message.text == "What is the capital of France?"

    def test_assistant(self):
        assert Message.assistant("Paris.").role is Role.ASSISTANT

    def test_assistant_may_be_empty_when_it_only_calls_tools(self):
        message = Message.assistant(
            tool_calls=(ToolCall(id="c1", name="search", arguments={"q": "x"}),)
        )

        assert message.text == ""
        assert message.tool_calls[0].name == "search"

    def test_tool(self):
        message = Message.tool([ToolResult(call_id="c1", name="search", content="Paris")])

        assert message.role is Role.TOOL
        assert message.tool_results[0].content == "Paris"


class TestContentParts:
    def test_text_property_concatenates_text_parts(self):
        message = Message(
            role=Role.ASSISTANT,
            content=(TextPart(text="Hello "), TextPart(text="world")),
        )

        assert message.text == "Hello world"

    def test_reasoning_is_not_part_of_the_answer(self):
        # A UI that renders `.text` must never surface the model's scratch work as
        # if it were the answer.
        message = Message(
            role=Role.ASSISTANT,
            content=(ReasoningPart(text="Let me think..."), TextPart(text="Paris.")),
        )

        assert message.text == "Paris."
        assert message.reasoning == "Let me think..."

    def test_images_are_reachable(self):
        message = Message(
            role=Role.USER,
            content=(TextPart(text="What is this?"), ImagePart(url="http://x/y.png")),
        )

        assert len(message.images) == 1
        assert message.images[0].url == "http://x/y.png"

    def test_parts_round_trip_through_json(self):
        # The discriminator is what makes stored history reloadable.
        original = Message(
            role=Role.ASSISTANT,
            content=(TextPart(text="hi"), ReasoningPart(text="hmm", signature="sig")),
        )

        restored = Message.model_validate_json(original.model_dump_json())

        assert restored == original
        assert isinstance(restored.content[1], ReasoningPart)

    def test_an_image_needs_exactly_one_source(self):
        with pytest.raises(ValidationError, match="exactly one"):
            ImagePart()

        with pytest.raises(ValidationError, match="exactly one"):
            ImagePart(data=b"x", url="http://x")


class TestRoleInvariants:
    def test_only_an_assistant_may_call_tools(self):
        with pytest.raises(ValidationError, match="assistant"):
            Message(
                role=Role.USER,
                content=(TextPart(text="hi"),),
                tool_calls=(ToolCall(id="c1", name="search"),),
            )

    def test_only_a_tool_message_may_carry_results(self):
        with pytest.raises(ValidationError, match="tool message"):
            Message(
                role=Role.ASSISTANT,
                content=(TextPart(text="hi"),),
                tool_results=(ToolResult(call_id="c1", name="search", content="x"),),
            )

    def test_a_tool_message_must_carry_a_result(self):
        with pytest.raises(ValidationError, match="at least one result"):
            Message(role=Role.TOOL)

    def test_a_user_message_must_have_content(self):
        with pytest.raises(ValidationError, match="must have content"):
            Message(role=Role.USER)

    def test_one_tool_message_carries_many_results(self):
        # Models emit parallel tool calls. One model turn, one response turn.
        message = Message.tool(
            [
                ToolResult(call_id="c1", name="search", content="a"),
                ToolResult(call_id="c2", name="fetch", content="b"),
            ]
        )

        assert len(message.tool_results) == 2


class TestToolCall:
    def test_malformed_arguments_are_carried_not_discarded(self):
        # Raising would turn a recoverable event into a dead request; coercing to
        # {} would invent an intent the model never had.
        call = ToolCall(id="c1", name="search", malformed_arguments='{"q": "unclosed')

        assert call.is_malformed
        assert call.arguments == {}

    def test_a_well_formed_call_is_not_malformed(self):
        assert not ToolCall(id="c1", name="search", arguments={"q": "x"}).is_malformed


class TestUsage:
    def test_total_tokens(self):
        assert Usage(input_tokens=100, output_tokens=50).total_tokens == 150

    def test_usage_sums_across_calls(self):
        # An agent run makes many calls; the number anyone wants is the run total.
        total = Usage(input_tokens=100, output_tokens=50, cached_input_tokens=10) + Usage(
            input_tokens=200, output_tokens=80, reasoning_tokens=40
        )

        assert total.input_tokens == 300
        assert total.output_tokens == 130
        assert total.cached_input_tokens == 10
        assert total.reasoning_tokens == 40

    def test_summing_an_empty_run_is_zero(self):
        assert sum((Usage(), Usage()), Usage()).total_tokens == 0

    def test_negative_tokens_are_refused(self):
        with pytest.raises(ValidationError):
            Usage(input_tokens=-1)


class TestImmutability:
    def test_messages_are_frozen(self):
        message = Message.user("hi")

        with pytest.raises(ValidationError):
            message.role = Role.ASSISTANT  # type: ignore[misc]

    def test_evolve_produces_a_new_message(self):
        original = Message.user("hi")

        changed = original.evolve(name="alice")

        assert changed.name == "alice"
        assert original.name is None  # untouched


def test_a_truncated_answer_is_detectable():
    # FinishReason.LENGTH means the answer is cut off. It is usually a bug, not an
    # answer, so callers must be able to tell the two apart.
    assert FinishReason("length") is FinishReason.LENGTH
    assert FinishReason.LENGTH.value == "length"
