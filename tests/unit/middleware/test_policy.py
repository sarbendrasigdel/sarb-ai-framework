"""Tests for the retry decision and the backoff curve.

Pure functions, so these are exhaustive and instant. Nothing here sleeps — a test that
asserts a thirty-second backoff by waiting thirty seconds is a test nobody runs.
"""

from __future__ import annotations

import random

import pytest

from sarb.config import ResilienceSettings
from sarb.errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidInputError,
    LLMError,
    ProviderTimeoutError,
    RateLimitError,
)
from sarb.middleware.policy import backoff_for, should_retry

POLICY = ResilienceSettings(max_attempts=3, initial_backoff=1.0, max_backoff=30.0)


class TestShouldRetry:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (RateLimitError("slow down", provider="p"), True),
            (ProviderTimeoutError("timed out", provider="p"), True),
            (LLMError("overloaded", provider="p", retryable=True), True),
            (AuthenticationError("bad key", provider="p"), False),
            (InvalidInputError("no such feature"), False),
            (LLMError("bad request", provider="p"), False),
        ],
    )
    def test_the_error_decides_not_us(self, error, expected):
        # This function never looks at a status code, a vendor exception, or a message.
        # It reads a verdict the adapter already reached — which is exactly what lets it
        # work on a provider nobody has written yet, including one from a plugin.
        assert should_retry(error, attempt=1, policy=POLICY) is expected

    def test_attempts_run_out(self):
        error = RateLimitError("slow down", provider="p")

        assert should_retry(error, attempt=2, policy=POLICY)
        assert not should_retry(error, attempt=3, policy=POLICY)

    def test_max_attempts_of_one_means_no_retry_at_all(self):
        # The off switch, and it has to be honoured for a judge or an eval run, where
        # three attempts is three times the bill.
        off = ResilienceSettings(max_attempts=1)

        assert not should_retry(RateLimitError("slow", provider="p"), attempt=1, policy=off)

    def test_a_bug_is_not_retried(self):
        # A TypeError from inside an adapter is a bug. Retrying it three times only
        # means it takes three times as long to find out.
        assert not should_retry(TypeError("adapter is broken"), attempt=1, policy=POLICY)

    def test_a_framework_error_that_is_not_a_provider_error_is_not_retried(self):
        # Bad config does not get better on the second attempt.
        assert not should_retry(ConfigurationError("bad yaml"), attempt=1, policy=POLICY)


class TestBackoff:
    def test_a_provider_that_says_when_is_believed(self):
        # It is the only party that knows when the quota resets. Guessing shorter means
        # retrying into the same rate limit and being told again, having spent an
        # attempt to learn nothing.
        error = RateLimitError("slow down", retry_after=42.0, provider="p")

        assert backoff_for(error, attempt=1, policy=POLICY) == 42.0

    def test_retry_after_is_not_capped_by_max_backoff(self):
        # `max_backoff` bounds *our* guessing. It does not get to overrule an
        # instruction from the provider.
        error = RateLimitError("slow down", retry_after=300.0, provider="p")

        assert backoff_for(error, attempt=1, policy=POLICY) == 300.0

    def test_the_curve_is_exponential(self):
        # Asserted against a rigged generator that always returns its upper bound, so
        # the shape of the curve is visible rather than obscured by the jitter.
        ceiling = _AlwaysMax()
        error = LLMError("boom", provider="p", retryable=True)

        waits = [backoff_for(error, attempt=n, policy=POLICY, rng=ceiling) for n in (1, 2, 3, 4)]

        assert waits == [1.0, 2.0, 4.0, 8.0]

    def test_the_curve_is_capped(self):
        ceiling = _AlwaysMax()
        error = LLMError("boom", provider="p", retryable=True)

        assert backoff_for(error, attempt=20, policy=POLICY, rng=ceiling) == 30.0

    def test_jitter_spreads_the_retries_across_the_window(self):
        # Not decoration. When an indexing job fires forty concurrent embedding batches
        # and all forty are rate limited in the same instant, an unjittered backoff
        # retries all forty at exactly t+1s — recreating the burst that caused the
        # problem, in lockstep, until the attempts run out.
        error = LLMError("boom", provider="p", retryable=True)

        waits = [
            backoff_for(error, attempt=3, policy=POLICY, rng=random.Random(seed))
            for seed in range(50)
        ]

        assert all(0.0 <= wait <= 4.0 for wait in waits)
        assert len(set(waits)) > 40  # they land in different places, which is the point


class _AlwaysMax(random.Random):
    """A generator with the jitter turned off, so the curve underneath is testable."""

    def uniform(self, a: float, b: float) -> float:
        return b
