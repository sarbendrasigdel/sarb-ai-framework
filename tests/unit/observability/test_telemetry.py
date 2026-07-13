"""Tests for tracing and metrics, and for the shared redaction rules."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from pydantic import SecretStr

from sarb.config import TelemetrySettings, load_settings
from sarb.contracts import Meter, Tracer
from sarb.errors import MissingDependencyError
from sarb.observability import configure_observability, get_logger, get_meter, get_tracer
from sarb.observability.telemetry import (
    NoOpMeter,
    NoOpTracer,
    configure_telemetry,
    reset_telemetry,
)
from sarb.observability.testing import capture_logs
from sarb.redaction import REDACTED, is_secret_key, redact


@pytest.fixture(autouse=True)
def _restore_telemetry():
    """Telemetry providers are process-global; do not leak between tests."""
    yield
    reset_telemetry()


class TestDefaults:
    def test_tracing_is_a_noop_until_configured(self):
        assert isinstance(get_tracer(), NoOpTracer)
        assert isinstance(get_meter(), NoOpMeter)

    def test_the_noop_tracer_still_yields_a_usable_span(self):
        # Instrumented code must not need to check whether telemetry is on.
        with get_tracer().span("llm.generate", model="claude-sonnet-5") as span:
            span.set_attribute("tokens", 42)
            span.add_event("first token")
            span.record_exception(ValueError("ignored"))

    def test_the_noop_meter_accepts_measurements(self):
        meter = get_meter()
        meter.counter("llm.tokens", 812, unit="token", model="claude-sonnet-5")
        meter.histogram("llm.latency", 0.42, unit="s")

    def test_an_exception_inside_a_noop_span_still_propagates(self):
        with pytest.raises(ValueError, match="boom"), get_tracer().span("op"):
            raise ValueError("boom")

    def test_disabling_telemetry_installs_the_noops(self):
        configure_telemetry(TelemetrySettings(enabled=False))

        assert isinstance(get_tracer(), NoOpTracer)

    def test_the_noops_satisfy_their_ports(self):
        assert isinstance(NoOpTracer(), Tracer)
        assert isinstance(NoOpMeter(), Meter)


def _otel_installed() -> bool:
    return importlib.util.find_spec("opentelemetry") is not None


class TestEnabledWithoutTheExtra:
    @pytest.mark.skipif(_otel_installed(), reason="requires opentelemetry to be absent")
    def test_enabling_telemetry_without_the_extra_names_the_install_command(self):
        # A process that silently discards the traces its operator asked for is
        # worse than one that refuses to start.
        with pytest.raises(MissingDependencyError) as exc_info:
            configure_telemetry(TelemetrySettings(enabled=True, endpoint="http://c:4318"))

        assert "sarb[telemetry]" in str(exc_info.value)


@pytest.mark.skipif(not _otel_installed(), reason="requires sarb[telemetry]")
class TestOpenTelemetry:
    def test_enabling_installs_a_real_tracer_and_meter(self):
        configure_telemetry(
            TelemetrySettings(enabled=True, endpoint="http://collector:4318", sample_ratio=1.0)
        )

        assert not isinstance(get_tracer(), NoOpTracer)
        assert isinstance(get_tracer(), Tracer)
        assert isinstance(get_meter(), Meter)

    def test_a_span_binds_trace_ids_into_the_log_context(self):
        # This is what makes "show me the logs for this trace" a query rather than
        # an exercise in timestamp arithmetic.
        configure_telemetry(
            TelemetrySettings(enabled=True, endpoint="http://collector:4318", sample_ratio=1.0)
        )

        with capture_logs() as records, get_tracer().span("llm.generate", model="x"):
            get_logger("test").info("inside the span")

        record = records.first("inside the span")
        assert len(record["trace_id"]) == 32
        assert len(record["span_id"]) == 16

    def test_trace_ids_do_not_leak_outside_the_span(self):
        configure_telemetry(
            TelemetrySettings(enabled=True, endpoint="http://collector:4318", sample_ratio=1.0)
        )

        with capture_logs() as records:
            with get_tracer().span("op"):
                get_logger("test").info("inside")
            get_logger("test").info("outside")

        assert "trace_id" in records.first("inside")
        assert "trace_id" not in records.first("outside")

    def test_a_raising_block_records_the_exception_and_re_raises(self):
        configure_telemetry(
            TelemetrySettings(enabled=True, endpoint="http://collector:4318", sample_ratio=1.0)
        )

        with pytest.raises(ValueError, match="boom"), get_tracer().span("op"):
            raise ValueError("boom")

    def test_measurements_are_accepted(self):
        configure_telemetry(
            TelemetrySettings(enabled=True, endpoint="http://collector:4318", sample_ratio=1.0)
        )

        meter = get_meter()
        meter.counter("llm.tokens", 812, unit="token", model="claude-sonnet-5")
        meter.counter("llm.tokens", 12, unit="token", model="claude-sonnet-5")  # cached instrument
        meter.histogram("llm.latency", 0.42, unit="s")

    def test_exotic_attribute_values_are_coerced_not_raised(self):
        # Observability code must never be able to break the request it observes.
        configure_telemetry(
            TelemetrySettings(enabled=True, endpoint="http://collector:4318", sample_ratio=1.0)
        )

        with get_tracer().span("op", weird={"a": 1}) as span:
            span.set_attribute("also_weird", object())
            span.add_event("thing", payload=[1, 2, 3])


class TestSetup:
    def test_configure_observability_logs_the_effective_config(self):
        settings = load_settings(overrides={"app": {"name": "demo"}})

        with capture_logs() as records:
            configure_observability(settings)

        record = records.first("application configured")
        assert record["app"] == "demo"
        assert record["telemetry"] is False

    def test_the_startup_line_never_leaks_a_secret(self):
        settings = load_settings(
            overrides={
                "database": {"dsn": "postgresql://user:hunter2@db/app"},
                "llms": {
                    "default": {"provider": "anthropic", "options": {"api_key": "sk-ant-live"}}
                },
            }
        )

        with capture_logs() as records:
            configure_observability(settings)

        dumped = str(records.first("application configured"))
        assert "hunter2" not in dumped
        assert "sk-ant-live" not in dumped
        assert REDACTED in dumped


class TestRedactionRules:
    @pytest.mark.parametrize(
        "key",
        ["api_key", "API_KEY", "apiKey", "x-api-key", "auth_token", "password", "dsn"],
    )
    def test_credential_shaped_keys_are_detected(self, key: str):
        assert is_secret_key(key)

    @pytest.mark.parametrize("key", ["model", "temperature", "collection", "chunks"])
    def test_ordinary_keys_are_not(self, key: str):
        assert not is_secret_key(key)

    @pytest.mark.parametrize(
        "key",
        ["tokens", "input_tokens", "output_tokens", "token_count", "total_tokens", "keys"],
    )
    def test_token_counts_are_not_mistaken_for_credentials(self, key: str):
        # A substring rule would mask every one of these, destroying the most
        # important metric an LLM framework emits. This is the regression that
        # keeps the matcher honest.
        assert not is_secret_key(key)

    @pytest.mark.parametrize("key", ["token", "access_token", "refresh_token"])
    def test_actual_tokens_are_still_caught(self, key: str):
        assert is_secret_key(key)

    def test_redaction_recurses_through_nesting(self):
        result = redact({"outer": {"inner": [{"api_key": "leak"}]}})

        assert result == {"outer": {"inner": [{"api_key": REDACTED}]}}

    def test_secret_str_is_masked_under_any_key(self):
        assert redact({"harmless": SecretStr("leak")}) == {"harmless": REDACTED}

    def test_the_input_is_never_mutated(self):
        original = {"api_key": "leak"}
        redact(original)

        assert original == {"api_key": "leak"}

    def test_pathological_nesting_is_masked_not_emitted(self):
        # A cycle or absurd depth must not be rendered raw: it has not been
        # inspected, so it cannot be assumed safe.
        deep: dict[str, Any] = {}
        node = deep
        for _ in range(20):
            node["next"] = {}
            node = node["next"]
        node["api_key"] = "leak"

        assert "leak" not in str(redact(deep))
