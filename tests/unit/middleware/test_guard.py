"""Tests for the guard: limit, retry, record.

The two that matter most are ``test_a_stream_is_not_retried_once_it_has_committed`` and
``test_a_backing_off_call_does_not_hold_a_slot``. Both protect behaviour that is
invisible when it is right and very expensive when it is wrong — one duplicates tokens
into a user's browser, the other takes a component offline under exactly the load it
was supposed to survive.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import anyio
import pytest

from sarb.config import ResilienceSettings
from sarb.contracts import Usage
from sarb.errors import AuthenticationError, RateLimitError
from sarb.middleware.guard import Guard, Outcome


def guard(**policy) -> Guard:
    return Guard(
        component="llms.default",
        provider="fake",
        policy=ResilienceSettings(**{"initial_backoff": 0.001, **policy}),
    )


class Flaky:
    """Fails a scripted number of times, then works. Counts what it was asked."""

    def __init__(self, failures: int = 0, error: Exception | None = None):
        self._failures = failures
        self._error = error or RateLimitError("slow down", provider="fake")
        self.attempts = 0

    async def call(self) -> str:
        self.attempts += 1
        if self.attempts <= self._failures:
            raise self._error
        return "ok"

    async def stream(self) -> AsyncIterator[str]:
        self.attempts += 1
        if self.attempts <= self._failures:
            raise self._error
        yield "one"
        yield "two"


class TestRetry:
    async def test_it_retries_a_retryable_failure_and_gets_an_answer(self):
        flaky = Flaky(failures=2)

        assert await guard().run("chat", flaky.call, target="m") == "ok"
        assert flaky.attempts == 3

    async def test_it_gives_up_when_the_attempts_run_out(self):
        flaky = Flaky(failures=99)

        with pytest.raises(RateLimitError):
            await guard(max_attempts=3).run("chat", flaky.call, target="m")

        assert flaky.attempts == 3

    async def test_it_does_not_retry_what_cannot_be_fixed_by_retrying(self):
        # A rejected key is still a rejected key on the third attempt, and each one
        # costs a round trip and a log line.
        flaky = Flaky(failures=99, error=AuthenticationError("bad key", provider="fake"))

        with pytest.raises(AuthenticationError):
            await guard().run("chat", flaky.call, target="m")

        assert flaky.attempts == 1

    async def test_max_attempts_of_one_disables_retry(self):
        flaky = Flaky(failures=1)

        with pytest.raises(RateLimitError):
            await guard(max_attempts=1).run("chat", flaky.call, target="m")

        assert flaky.attempts == 1

    async def test_the_call_is_a_factory_because_a_coroutine_cannot_be_awaited_twice(self):
        # If `run` took the coroutine rather than a way of making one, the second
        # attempt would raise RuntimeError instead of retrying. This asserts the shape
        # of the API, which exists solely to make retry possible.
        flaky = Flaky(failures=1)

        await guard().run("chat", flaky.call, target="m")

        assert flaky.attempts == 2

    async def test_it_waits_for_as_long_as_the_provider_asked(self):
        slow = Flaky(failures=1, error=RateLimitError("slow", retry_after=0.05, provider="fake"))

        with anyio.move_on_after(0.02) as scope:
            await guard().run("chat", slow.call, target="m")

        assert scope.cancelled_caught, "the guard should still have been waiting"

    async def test_cancellation_is_never_retried_or_slept_through(self):
        # Swallowing a cancellation here would hang a shutdown for max_attempts times
        # max_backoff, and the process would be killed rather than closed.
        entered = anyio.Event()

        async def hangs() -> str:
            entered.set()
            await anyio.sleep(10)
            return "never"

        async with anyio.create_task_group() as group:
            group.start_soon(lambda: guard().run("chat", hangs, target="m"))
            await entered.wait()
            group.cancel_scope.cancel()

        # Reaching here at all is the assertion: the task group closed promptly rather
        # than waiting out three attempts and their backoff.


class TestStreaming:
    async def test_a_stream_yields_its_events(self):
        events = [e async for e in guard().stream("chat", Flaky().stream, target="m")]

        assert events == ["one", "two"]

    async def test_a_stream_is_retried_before_it_has_produced_anything(self):
        # A failure here has produced nothing anyone has seen, so another attempt is
        # invisible to the caller — and a 429 on a streaming endpoint should not be
        # fatal when it is fatal nowhere else.
        flaky = Flaky(failures=2)

        events = [e async for e in guard().stream("chat", flaky.stream, target="m")]

        assert events == ["one", "two"]
        assert flaky.attempts == 3

    async def test_a_stream_is_not_retried_once_it_has_committed(self):
        # The one that matters. Once a token has been handed to the caller it has been
        # rendered, logged, or written to a socket — and no retry can take it back.
        # Retrying here would restart the answer in the user's browser, mid-sentence.
        attempts = 0

        async def breaks_midway() -> AsyncIterator[str]:
            nonlocal attempts
            attempts += 1
            yield "one"
            raise RateLimitError("slow down", provider="fake")  # retryable, and yet

        async def drain() -> list[str]:
            return [e async for e in guard().stream("chat", breaks_midway, target="m")]

        with pytest.raises(RateLimitError):
            await drain()

        assert attempts == 1, "a committed stream must not be restarted"

    async def test_an_empty_stream_is_not_an_error(self):
        async def nothing() -> AsyncIterator[str]:
            return
            yield  # pragma: no cover - makes this an async generator

        assert [e async for e in guard().stream("chat", nothing, target="m")] == []

    async def test_a_stream_that_never_starts_gives_up_like_anything_else(self):
        flaky = Flaky(failures=99)

        with pytest.raises(RateLimitError):
            _ = [e async for e in guard(max_attempts=2).stream("chat", flaky.stream, target="m")]

        assert flaky.attempts == 2


class TestConcurrencyLimit:
    async def test_it_caps_calls_in_flight(self):
        # A hosted API will happily accept forty concurrent requests and rate limit all
        # forty. Queuing is cheaper than being throttled.
        limited = guard(max_concurrency=2)
        concurrent = 0
        peak = 0

        async def slow() -> str:
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            await anyio.sleep(0.01)
            concurrent -= 1
            return "ok"

        async with anyio.create_task_group() as group:
            for _ in range(8):
                group.start_soon(lambda: limited.run("chat", slow, target="m"))

        assert peak == 2

    async def test_a_backing_off_call_does_not_hold_a_slot(self):
        # Why the limiter is innermost. If it sat outside the retry, a rate-limited call
        # would keep its slot for the whole backoff sleep — so one throttled request
        # would block a healthy one from even starting, and a burst of 429s would take
        # the component offline rather than merely slowing it down.
        #
        # The backoff is pinned with an explicit Retry-After rather than left to the
        # jittered curve. `uniform(0, 0.05)` can return a millisecond, and then this
        # test would pass with the limiter in the *wrong* place — proving nothing, on
        # most runs, which is worse than not having it.
        limited = Guard(
            component="llms.default",
            provider="fake",
            policy=ResilienceSettings(max_attempts=3, max_concurrency=1),
        )
        backing_off = Flaky(
            failures=2,
            error=RateLimitError("slow down", retry_after=0.5, provider="fake"),
        )
        ran = anyio.Event()

        async def healthy() -> str:
            ran.set()
            return "ok"

        async with anyio.create_task_group() as group:
            group.start_soon(lambda: limited.run("chat", backing_off.call, target="m"))
            await anyio.sleep(0.01)  # let it fail once and settle into its backoff
            group.start_soon(lambda: limited.run("chat", healthy, target="m"))

            # There is exactly one slot. The healthy call can only get it if the
            # backing-off one handed it back before going to sleep — which is the whole
            # reason the limiter sits inside the retry rather than outside it.
            with anyio.fail_after(0.04):
                await ran.wait()

    async def test_unlimited_is_the_default(self):
        # The right answer for a local model that already queues internally, and the
        # wrong one for a hosted API — so it is a decision, made per component.
        unlimited = guard()
        concurrent = 0
        peak = 0

        async def slow() -> str:
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            await anyio.sleep(0.01)
            concurrent -= 1
            return "ok"

        async with anyio.create_task_group() as group:
            for _ in range(5):
                group.start_soon(lambda: unlimited.run("chat", slow, target="m"))

        assert peak == 5


class TestOutcome:
    async def test_the_wrapper_describes_its_own_result(self):
        # The guard cannot read an LLMResponse and must keep not knowing what one is —
        # the same guard wraps embedders, and will wrap vector stores. So the per-port
        # wrapper hands it an Outcome, and the guard records what it is given.
        seen: list[Outcome] = []

        async def call() -> str:
            return "done"

        def describe(result: str) -> Outcome:
            seen.append(Outcome(model="m-1", usage=Usage(input_tokens=10, output_tokens=2)))
            return seen[-1]

        await guard().run("chat", call, target="m", outcome=describe)

        assert seen[0].usage == Usage(input_tokens=10, output_tokens=2)
