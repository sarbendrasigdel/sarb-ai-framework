"""Tests for structured logging, field merging, and redaction.

The redaction tests are the ones that matter most: a leaked credential in a log
aggregator is a security incident, and it is the failure mode most likely to be
introduced by a well-meaning change to the formatter.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pytest
from pydantic import SecretStr

from sarb.config import LoggingSettings, LogLevel
from sarb.contracts import Logger
from sarb.errors import RateLimitError
from sarb.observability import configure_logging, get_logger, log_context
from sarb.observability.logging import ConsoleFormatter, JsonFormatter, StdlibLogger
from sarb.observability.testing import capture_logs
from sarb.redaction import REDACTED


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


def _lines(stream: io.StringIO) -> list[dict[str, Any]]:
    """Parse the JSON records written to a stream."""
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


class TestStructuredFields:
    def test_fields_become_json_keys(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        get_logger("test").info("llm call finished", model="claude-sonnet-5", tokens=812)

        (record,) = _lines(stream)
        assert record["event"] == "llm call finished"
        assert record["level"] == "INFO"
        assert record["model"] == "claude-sonnet-5"
        assert record["tokens"] == 812  # a number, not a string

    def test_record_carries_timestamp_and_logger_name(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        get_logger("sarb.retrieval").info("done")

        (record,) = _lines(stream)
        assert record["logger"] == "sarb.retrieval"
        assert record["timestamp"].endswith("+00:00")  # UTC, unambiguous

    def test_unserialisable_values_do_not_break_the_caller(self, stream: io.StringIO):
        # Logging is allowed to render a value badly. It is never allowed to fail
        # and take down the request it was trying to describe.
        configure_logging(LoggingSettings(), stream=stream)

        class Exotic:
            def __repr__(self) -> str:
                return "<exotic>"

        get_logger("test").info("weird", value=Exotic())

        (record,) = _lines(stream)
        assert record["value"] == "<exotic>"


class TestBinding:
    def test_bound_fields_appear_on_every_record(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        logger = get_logger("test").bind(run_id="r-1")
        logger.info("step one")
        logger.info("step two")

        assert all(record["run_id"] == "r-1" for record in _lines(stream))

    def test_bind_does_not_mutate_the_parent(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        parent = get_logger("test")
        parent.bind(child_only="x").info("child")
        parent.info("parent")

        child, parent_record = _lines(stream)
        assert child["child_only"] == "x"
        assert "child_only" not in parent_record  # no leak back upward

    def test_call_site_fields_win_over_bound_fields(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        get_logger("test").bind(model="default").info("call", model="override")

        assert _lines(stream)[0]["model"] == "override"


class TestLogContext:
    def test_context_fields_reach_every_record(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        with log_context(request_id="req-1"):
            get_logger("a").info("first")
            get_logger("b").info("second")

        assert all(record["request_id"] == "req-1" for record in _lines(stream))

    def test_context_nests_additively_and_unwinds(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        with log_context(request_id="req-1"):
            with log_context(agent="researcher"):
                get_logger("x").info("inner")
            get_logger("x").info("outer")

        inner, outer = _lines(stream)
        assert inner["request_id"] == "req-1"
        assert inner["agent"] == "researcher"
        assert outer["request_id"] == "req-1"
        assert "agent" not in outer  # inner field did not leak out

    def test_context_is_restored_even_when_the_block_raises(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        with pytest.raises(ValueError, match="boom"), log_context(request_id="req-1"):
            raise ValueError("boom")

        get_logger("x").info("after")

        assert "request_id" not in _lines(stream)[0]

    def test_bound_fields_win_over_context(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        with log_context(source="context"):
            get_logger("x").bind(source="bound").info("who wins")

        assert _lines(stream)[0]["source"] == "bound"


class TestRedaction:
    def test_secret_looking_fields_are_masked(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        get_logger("test").info("configured", model="claude-sonnet-5", api_key="sk-ant-live")

        record = _lines(stream)[0]
        assert record["api_key"] == REDACTED
        assert record["model"] == "claude-sonnet-5"
        assert "sk-ant-live" not in stream.getvalue()

    def test_secrets_nested_in_a_field_are_masked(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        get_logger("test").info("config", options={"model": "x", "api_key": "sk-live"})

        assert "sk-live" not in stream.getvalue()

    def test_secret_str_is_masked_whatever_the_key(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        get_logger("test").info("odd", innocuous_name=SecretStr("sk-live"))

        assert _lines(stream)[0]["innocuous_name"] == REDACTED
        assert "sk-live" not in stream.getvalue()

    def test_custom_redact_keys_are_honoured(self, stream: io.StringIO):
        configure_logging(LoggingSettings(redact_keys=("ssn",)), stream=stream)

        get_logger("test").info("pii", ssn="123-45-6789")

        assert _lines(stream)[0]["ssn"] == REDACTED

    def test_redaction_survives_in_captured_logs(self):
        # The test helper must not be a hole in the redaction guarantee.
        with capture_logs() as records:
            get_logger("test").info("configured", api_key="sk-live")

        assert records.first("configured")["api_key"] == REDACTED


class TestErrors:
    def test_exception_is_rendered(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        try:
            raise ValueError("bad input")
        except ValueError as exc:
            get_logger("test").error("failed", exc_info=exc)

        record = _lines(stream)[0]
        assert record["level"] == "ERROR"
        assert "ValueError: bad input" in record["exception"]
        assert record["error_type"] == "ValueError"

    def test_a_framework_errors_context_becomes_queryable_fields(self, stream: io.StringIO):
        # The point: `provider=openai` is a field you can filter on, not a
        # fragment of a message string you have to regex out.
        configure_logging(LoggingSettings(), stream=stream)

        exc = RateLimitError("slow down", provider="openai", retry_after=30.0)
        get_logger("test").error("llm call failed", exc_info=exc)

        record = _lines(stream)[0]
        assert record["provider"] == "openai"
        assert record["retry_after"] == 30.0
        assert record["error_type"] == "RateLimitError"

    def test_call_site_fields_win_over_error_context(self, stream: io.StringIO):
        configure_logging(LoggingSettings(), stream=stream)

        exc = RateLimitError("slow down", provider="openai")
        get_logger("test").error("failed", exc_info=exc, provider="explicit")

        assert _lines(stream)[0]["provider"] == "explicit"


class TestConfiguration:
    def test_level_is_honoured(self, stream: io.StringIO):
        configure_logging(LoggingSettings(level=LogLevel.WARNING), stream=stream)

        logger = get_logger("test")
        logger.info("invisible")
        logger.warning("visible")

        assert [r["event"] for r in _lines(stream)] == ["visible"]

    def test_console_format_is_human_readable(self, stream: io.StringIO):
        configure_logging(LoggingSettings(format="console"), stream=stream)

        get_logger("test").info("ready", port=8000)

        output = stream.getvalue()
        assert "ready" in output
        assert "port=8000" in output
        assert not output.startswith("{")  # not JSON

    def test_reconfiguring_does_not_double_emit(self, stream: io.StringIO):
        # Two handlers stacked means every line appears twice, which corrupts any
        # count-based alerting built on these logs.
        configure_logging(LoggingSettings(), stream=stream)
        configure_logging(LoggingSettings(), stream=stream)

        get_logger("test").info("once")

        assert len(_lines(stream)) == 1

    def test_third_party_extra_fields_are_still_captured(self, stream: io.StringIO):
        # A library that does `logging.info(..., extra={"status": 500})` should
        # still produce a structured field, not lose it.
        configure_logging(LoggingSettings(), stream=stream)

        logging.getLogger("some.library").warning("upstream slow", extra={"status": 503})

        assert _lines(stream)[0]["status"] == 503


class TestPortConformance:
    def test_the_default_logger_satisfies_the_port(self):
        assert isinstance(get_logger("test"), Logger)

    def test_a_bound_logger_is_still_a_logger(self):
        assert isinstance(get_logger("test").bind(x=1), Logger)

    def test_a_disabled_level_skips_the_field_merge(self):
        # DEBUG logging in a hot retrieval loop must cost approximately nothing.
        logger = StdlibLogger(logging.getLogger("quiet"))
        logging.getLogger("quiet").setLevel(logging.ERROR)

        calls = 0

        class Counting:
            def __repr__(self) -> str:
                nonlocal calls
                calls += 1
                return "expensive"

        logger.debug("skipped", value=Counting())

        assert calls == 0


class TestFormattersDirectly:
    def test_json_formatter_emits_one_line(self):
        formatter = JsonFormatter(frozenset({"key"}))
        record = logging.LogRecord("t", logging.INFO, "", 0, "event", None, None)

        output = formatter.format(record)

        assert "\n" not in output
        assert json.loads(output)["event"] == "event"

    def test_console_formatter_without_a_tty_emits_no_colour(self):
        formatter = ConsoleFormatter(frozenset(), colour=False)
        record = logging.LogRecord("t", logging.INFO, "", 0, "event", None, None)

        assert "\033" not in formatter.format(record)
