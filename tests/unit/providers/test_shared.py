"""Tests for the machinery every adapter shares, whatever family it is in.

The status table lives one level above the families because a 429 means the same
thing whether it interrupted a chat completion or a batch of embeddings. These tests
pin that down: the *kind* of call is the caller's knowledge, and the *meaning* of the
status is not.
"""

from __future__ import annotations

import pytest

from sarb.errors import (
    AuthenticationError,
    EmbeddingError,
    InvalidInputError,
    LLMError,
    ProviderTimeoutError,
    RateLimitError,
)
from sarb.providers._shared import error_for_status, retry_after_seconds, unsupported


class TestErrorForStatus:
    @pytest.mark.parametrize(
        ("status", "expected", "retryable"),
        [
            (401, AuthenticationError, False),
            (403, AuthenticationError, False),
            (429, RateLimitError, True),
            (408, ProviderTimeoutError, True),
            (504, ProviderTimeoutError, True),
            (500, LLMError, True),
            (503, LLMError, True),
            (529, LLMError, True),  # Anthropic's "overloaded"
            (400, LLMError, False),
            (404, LLMError, False),
            (None, LLMError, False),
        ],
    )
    def test_status_decides_the_class_and_the_retry_semantics(self, status, expected, retryable):
        error = error_for_status(
            status, "boom", provider="anthropic", model="m", otherwise=LLMError
        )

        assert isinstance(error, expected)
        assert error.retryable is retryable

    def test_the_model_is_queryable(self):
        # The context lands in log fields, so "every rate limit on gpt-4o" is a query
        # rather than a grep.
        error = error_for_status(
            429, "slow down", provider="openai", model="gpt-4o", otherwise=LLMError
        )

        assert error.context["model"] == "gpt-4o"
        assert error.context["provider"] == "openai"

    def test_retry_after_is_carried(self):
        error = error_for_status(
            429, "slow down", provider="openai", model="m", otherwise=LLMError, retry_after=30.0
        )

        assert isinstance(error, RateLimitError)
        assert error.retry_after == 30.0

    def test_the_fallback_class_is_the_callers_to_choose(self):
        # The same 500 is an LLMError from a chat call and an EmbeddingError from an
        # embedding one — but it is retryable in both, and that is the part the
        # resilience layer reads. Splitting the table per family would let those two
        # answers drift apart.
        chat = error_for_status(500, "boom", provider="openai", model="m", otherwise=LLMError)
        embed = error_for_status(
            500, "boom", provider="openai", model="m", otherwise=EmbeddingError
        )

        assert isinstance(chat, LLMError)
        assert isinstance(embed, EmbeddingError)
        assert chat.retryable
        assert embed.retryable

    def test_a_rate_limit_is_a_rate_limit_whatever_the_family(self):
        # RateLimitError is not parameterised by family, deliberately: a backoff
        # policy that had to ask "which kind of 429 was that?" would be the wrong
        # abstraction.
        error = error_for_status(
            429, "slow", provider="openai", model="m", otherwise=EmbeddingError
        )

        assert isinstance(error, RateLimitError)


class TestRetryAfter:
    def test_reads_the_header_case_insensitively(self):
        assert retry_after_seconds({"Retry-After": "30"}) == 30.0
        assert retry_after_seconds({"retry-after": "30"}) == 30.0

    @pytest.mark.parametrize(
        "headers",
        [None, {}, {"x-other": "1"}, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}],
    )
    def test_anything_it_cannot_read_is_no_answer_rather_than_a_guess(self, headers):
        # The HTTP-date form is legal. Guessing at it would produce a number the
        # resilience layer *prefers* over its own backoff — so a bad parse hammers a
        # provider that just asked to be left alone.
        assert retry_after_seconds(headers) is None


def test_unsupported_names_the_provider_and_the_feature():
    error = unsupported("ollama", 'tool_choice mode "required"', hint="Ollama supports auto only.")

    assert isinstance(error, InvalidInputError)
    assert error.context == {"provider": "ollama", "feature": 'tool_choice mode "required"'}
    assert "Ollama supports auto only." in str(error)
