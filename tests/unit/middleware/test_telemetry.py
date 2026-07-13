"""What the guard records, asserted rather than assumed.

Every adapter in this framework has been carefully returning token counts since Module
5, and until now nothing read them. These tests are the proof that something finally
does — and the protection against it silently stopping, which is a failure that
produces no exception and no red test, only an empty dashboard on the day it is needed.
"""

from __future__ import annotations

import pytest

from sarb.config import ModelPrice, ResilienceSettings
from sarb.contracts import LLMRequest, Message
from sarb.errors import AuthenticationError, RateLimitError
from sarb.middleware import semconv
from sarb.middleware.guard import Guard
from sarb.middleware.llm import GuardedLLM
from sarb.providers.llm.fake import FakeOptions, FakeProvider

from .recording import RecordingLogger, install, named

#: $1 per million in, $2 per million out. Round numbers, so an assertion on a cost is
#: readable as arithmetic rather than as a magic constant.
PRICING = {"fake": {"fake-1": ModelPrice(input=1.0, output=2.0)}}


def guarded(
    *,
    policy: ResilienceSettings | None = None,
    pricing: dict[str, dict[str, ModelPrice]] | None = None,
    **options,
) -> GuardedLLM:
    return GuardedLLM(
        FakeProvider(FakeOptions(model="fake-1", **options)),
        Guard(
            component="llms.judge",
            provider="fake",
            policy=policy or ResilienceSettings(max_attempts=3, initial_backoff=0.001),
            pricing=PRICING if pricing is None else pricing,
        ),
    )


# Long enough to cost a token. The fake bills a prompt at roughly four characters per
# token, so "hi" genuinely costs zero — and the guard, correctly, does not record a
# zero. A test that asserted on the token count of a two-character prompt would be
# asserting on nothing.
ASK = LLMRequest(messages=(Message.user("What is the capital of France, and why?"),))


class TestSpans:
    async def test_a_call_opens_one_span_named_for_the_operation_and_model(self, monkeypatch):
        tracer, _ = install(monkeypatch)

        await guarded().generate(ASK)

        [span] = tracer.spans
        assert span.name == "chat fake-1"
        assert span.attributes[semconv.OPERATION] == "chat"
        assert span.attributes[semconv.PROVIDER] == "fake"
        assert span.attributes[semconv.REQUEST_MODEL] == "fake-1"

    async def test_the_component_name_is_on_the_span(self, monkeypatch):
        # The attribute anyone actually groups by. An app runs three models and two are
        # from the same vendor; `gen_ai.provider.name` cannot tell them apart, and the
        # config already gave them names.
        tracer, _ = install(monkeypatch)

        await guarded().generate(ASK)

        assert tracer.spans[0].attributes[semconv.COMPONENT] == "llms.judge"

    async def test_token_usage_lands_on_the_span(self, monkeypatch):
        tracer, _ = install(monkeypatch)

        await guarded(responses=("a long enough answer to cost something",)).generate(ASK)

        attributes = tracer.spans[0].attributes
        assert attributes[semconv.INPUT_TOKENS] > 0
        assert attributes[semconv.OUTPUT_TOKENS] > 0

    async def test_retries_share_one_span(self, monkeypatch):
        # One logical call, one span. A span per attempt would make a trace unreadable
        # and would report the duration of the lucky attempt rather than the wait the
        # caller actually endured.
        tracer, _ = install(monkeypatch)

        await guarded(fail_with="rate_limit", fail_times=2).generate(ASK)

        assert len(tracer.spans) == 1

    async def test_a_failure_is_recorded_on_the_span(self, monkeypatch):
        tracer, _ = install(monkeypatch)

        with pytest.raises(AuthenticationError):
            await guarded(fail_with="auth").generate(ASK)

        assert isinstance(tracer.spans[0].exceptions[0], AuthenticationError)


class TestMetrics:
    async def test_duration_is_recorded(self, monkeypatch):
        _, meter = install(monkeypatch)

        await guarded().generate(ASK)

        [duration] = named(meter.histograms, semconv.DURATION)
        assert duration.value >= 0
        assert duration.attributes[semconv.COMPONENT] == "llms.judge"

    async def test_duration_is_what_the_caller_waited_not_what_the_last_attempt_took(
        self, monkeypatch
    ):
        # A histogram that timed only the attempt that happened to succeed would report
        # a healthy p99 for a system that is, from the outside, unusable.
        _, meter = install(monkeypatch)
        slow = guarded(
            fail_with="rate_limit",
            fail_times=2,
            policy=ResilienceSettings(max_attempts=3, initial_backoff=0.02),
        )

        await slow.generate(ASK)

        [duration] = named(meter.histograms, semconv.DURATION)
        assert duration.value > 0.02, "the backoff is part of what the caller waited"

    async def test_tokens_are_split_by_direction(self, monkeypatch):
        _, meter = install(monkeypatch)

        await guarded(responses=("a long enough answer to cost something",)).generate(ASK)

        tokens = named(meter.histograms, semconv.TOKENS)
        assert {m.attributes[semconv.TOKEN_TYPE] for m in tokens} == {"input", "output"}

    async def test_retries_are_counted(self, monkeypatch):
        # The earliest warning there is: retries climb well before failures do.
        _, meter = install(monkeypatch)

        await guarded(fail_with="rate_limit", fail_times=2).generate(ASK)

        retries = named(meter.counters, semconv.RETRIES)
        assert len(retries) == 2
        assert retries[0].attributes[semconv.ERROR_TYPE] == "RateLimitError"

    async def test_a_successful_call_counts_no_retries(self, monkeypatch):
        _, meter = install(monkeypatch)

        await guarded().generate(ASK)

        assert named(meter.counters, semconv.RETRIES) == []

    async def test_a_failure_is_tagged_with_the_error_class_and_not_its_message(self, monkeypatch):
        # A message is unbounded — it carries ids, model names, sometimes whole prompts.
        # An unbounded metric dimension is how a metrics backend falls over.
        _, meter = install(monkeypatch)

        with pytest.raises(RateLimitError):
            await guarded(
                fail_with="rate_limit", policy=ResilienceSettings(max_attempts=1)
            ).generate(ASK)

        [duration] = named(meter.histograms, semconv.DURATION)
        assert duration.attributes[semconv.ERROR_TYPE] == "RateLimitError"
        assert "slow" not in str(duration.attributes)


class TestStreamingIsRecordedToo:
    async def test_a_stream_gets_a_span(self, monkeypatch):
        tracer, meter = install(monkeypatch)

        _ = [event async for event in guarded().stream(ASK)]

        assert tracer.spans[0].name == "chat fake-1"
        assert len(named(meter.histograms, semconv.DURATION)) == 1

    async def test_a_stream_records_its_tokens(self, monkeypatch):
        # The regression test for the largest hole this framework has had. The guard
        # recorded a span and a duration for a stream and *no tokens whatsoever*, because
        # `Guard.stream` had no outcome hook to read them with.
        #
        # It was invisible precisely because it was not an error: the graphs existed, the
        # spans existed, the numbers on them were simply absent. And since anything with
        # a chat interface streams, the missing numbers were most of the bill.
        _, meter = install(monkeypatch)
        answer = "a long enough answer to cost a measurable number of tokens"

        _ = [event async for event in guarded(responses=(answer,)).stream(ASK)]

        tokens = named(meter.histograms, semconv.TOKENS)
        assert {m.attributes[semconv.TOKEN_TYPE] for m in tokens} == {"input", "output"}

    async def test_a_stream_costs_the_same_as_the_call_it_streams(self, monkeypatch):
        # Whether the caller asked for the answer all at once or a token at a time is a
        # detail of delivery. The invoice does not care, and neither may the metric.
        _, meter = install(monkeypatch)
        answer = "a long enough answer to cost a measurable number of tokens"

        await guarded(responses=(answer,)).generate(ASK)
        [unary] = named(meter.counters, semconv.COST)

        _, meter = install(monkeypatch)
        _ = [event async for event in guarded(responses=(answer,)).stream(ASK)]
        [streamed] = named(meter.counters, semconv.COST)

        assert streamed.value == unary.value


class TestCost:
    async def test_a_priced_call_records_dollars(self, monkeypatch):
        _, meter = install(monkeypatch)

        await guarded(responses=("a long enough answer to cost something",)).generate(ASK)

        [cost] = named(meter.counters, semconv.COST)
        [tokens_in] = [
            m
            for m in named(meter.histograms, semconv.TOKENS)
            if m.attributes[semconv.TOKEN_TYPE] == "input"
        ]
        [tokens_out] = [
            m
            for m in named(meter.histograms, semconv.TOKENS)
            if m.attributes[semconv.TOKEN_TYPE] == "output"
        ]

        # $1/M in and $2/M out, so the cost is exactly the arithmetic and not a constant
        # somebody once observed and pasted in.
        expected = (tokens_in.value * 1.0 + tokens_out.value * 2.0) / 1_000_000
        assert cost.value == pytest.approx(expected)
        assert cost.attributes[semconv.COMPONENT] == "llms.judge"

    async def test_an_unpriced_model_records_tokens_but_no_cost(self, monkeypatch):
        # The tokens are still true. Only the dollars are unknown, and a guessed zero
        # would be a lie that sums.
        _, meter = install(monkeypatch)

        await guarded(responses=("a long enough answer to cost something",), pricing={}).generate(
            ASK
        )

        assert named(meter.counters, semconv.COST) == []
        assert named(meter.histograms, semconv.TOKENS) != []

    async def test_it_complains_about_an_unpriced_model_once_and_not_once_a_call(self, monkeypatch):
        # Once, because a warning on every call is a warning everybody filters out of
        # their logs — and at all, because the alternative is a cost dashboard with no bar
        # for this model, which looks exactly like a model nobody is using.
        install(monkeypatch)
        llm = guarded(pricing={})
        log = RecordingLogger()
        monkeypatch.setattr(llm._guard, "_log", log)

        for _ in range(3):
            await llm.generate(ASK)

        assert len(log.warnings) == 1
        assert log.warnings[0]["model"] == "fake-1"
        assert "pricing.fake.fake-1" in log.warnings[0]["hint"]
