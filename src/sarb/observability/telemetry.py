"""Tracing and metrics, no-op by default, OpenTelemetry when configured.

The default really is a no-op — not a "local exporter", not a buffer. A framework
that costs something to *not* use its telemetry is a framework people rip the
telemetry out of. ``with tracer.span(...)`` on the default tracer allocates one
object and calls no methods.

Turning it on is configuration, not code:

    telemetry:
      enabled: true
      endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT}
      sample_ratio: 0.1

Correlation
-----------
When OpenTelemetry is active, opening a span binds ``trace_id`` and ``span_id``
into the logging context, so every log line emitted inside the span carries them.
That is what makes "show me the logs for this trace" a query rather than an
exercise in timestamp arithmetic — and it is the entire reason the two subsystems
are in one module instead of two.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sarb.config import TelemetrySettings
from sarb.contracts.observability import Meter, Span, Tracer
from sarb.dependencies import import_optional
from sarb.observability.context import log_context

__all__ = [
    "NoOpMeter",
    "NoOpSpan",
    "NoOpTracer",
    "configure_telemetry",
    "get_meter",
    "get_tracer",
]


# --------------------------------------------------------------------------- #
# The default: nothing at all
# --------------------------------------------------------------------------- #


class NoOpSpan:
    """A span that records nothing."""

    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        """Discard the attribute."""

    def add_event(self, name: str, **attributes: Any) -> None:
        """Discard the event."""

    def record_exception(self, exception: BaseException) -> None:
        """Discard the exception; it still propagates to the caller."""


class NoOpTracer:
    """A tracer that creates no spans. The default."""

    __slots__ = ()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:  # noqa: ARG002
        """Yield a span that discards everything.

        The arguments are ignored on purpose: this is the whole point of the
        no-op, and instrumented code must never have to check whether telemetry
        happens to be switched on.
        """
        yield _NO_OP_SPAN


class NoOpMeter:
    """A meter that records nothing. The default."""

    __slots__ = ()

    def counter(
        self, name: str, value: int | float = 1, *, unit: str = "", **attributes: Any
    ) -> None:
        """Discard the measurement."""

    def histogram(self, name: str, value: float, *, unit: str = "", **attributes: Any) -> None:
        """Discard the measurement."""


_NO_OP_SPAN = NoOpSpan()

# Process-global providers. Set once by configure_telemetry() at the composition
# root. Global rather than injected, deliberately: threading a tracer through
# every constructor in the framework would be pure ceremony, and it is the model
# both stdlib logging and OpenTelemetry itself converged on.
_tracer: Tracer = NoOpTracer()
_meter: Meter = NoOpMeter()


def get_tracer() -> Tracer:
    """The active tracer. A no-op unless telemetry is configured."""
    return _tracer


def get_meter() -> Meter:
    """The active meter. A no-op unless telemetry is configured."""
    return _meter


# --------------------------------------------------------------------------- #
# OpenTelemetry
# --------------------------------------------------------------------------- #


class OTelSpan:
    """Adapts an OpenTelemetry span to the :class:`Span` port."""

    __slots__ = ("_span",)

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        """Set one attribute, coercing values OTel will not accept."""
        self._span.set_attribute(key, _coerce(value))

    def add_event(self, name: str, **attributes: Any) -> None:
        """Record a point-in-time event."""
        self._span.add_event(name, attributes=_attributes(attributes))

    def record_exception(self, exception: BaseException) -> None:
        """Attach an exception without ending the span."""
        self._span.record_exception(exception)


class OTelTracer:
    """Adapts an OpenTelemetry tracer to the :class:`Tracer` port."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        """Open a span, bind its ids to the log context, and record failures.

        A raising block marks the span as errored and records the exception before
        re-raising. Doing it here rather than at every call site means it cannot be
        forgotten — and the one span somebody forgets is always the one that
        mattered.
        """
        trace_api = import_optional("opentelemetry.trace", extra="telemetry")

        with self._tracer.start_as_current_span(
            name, attributes=_attributes(attributes)
        ) as raw_span:
            context = raw_span.get_span_context()
            with log_context(
                trace_id=format(context.trace_id, "032x"),
                span_id=format(context.span_id, "016x"),
            ):
                try:
                    yield OTelSpan(raw_span)
                except Exception as exc:
                    raw_span.record_exception(exc)
                    raw_span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exc)))
                    raise


class OTelMeter:
    """Adapts an OpenTelemetry meter to the :class:`Meter` port.

    Instruments are created once and cached by name. OTel expects an instrument to
    be a long-lived object; creating one per measurement is both wasteful and, in
    some SDK versions, a source of duplicate-registration warnings.
    """

    __slots__ = ("_counters", "_histograms", "_meter")

    def __init__(self, meter: Any) -> None:
        self._meter = meter
        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}

    def counter(
        self, name: str, value: int | float = 1, *, unit: str = "", **attributes: Any
    ) -> None:
        """Add to a monotonic counter."""
        instrument = self._counters.get(name)
        if instrument is None:
            instrument = self._meter.create_counter(name, unit=unit)
            self._counters[name] = instrument
        instrument.add(value, attributes=_attributes(attributes))

    def histogram(self, name: str, value: float, *, unit: str = "", **attributes: Any) -> None:
        """Record a distribution sample."""
        instrument = self._histograms.get(name)
        if instrument is None:
            instrument = self._meter.create_histogram(name, unit=unit)
            self._histograms[name] = instrument
        instrument.record(value, attributes=_attributes(attributes))


def configure_telemetry(settings: TelemetrySettings) -> None:
    """Install the tracer and meter the settings ask for.

    Does nothing when ``telemetry.enabled`` is false, which is the default and the
    right default: a developer running tests should not need a collector.

    Args:
        settings: The ``telemetry`` section of the application settings.

    Raises:
        MissingDependencyError: If telemetry is enabled without the extra
            installed. Enabled-but-not-installed is a misconfiguration, and a
            process that silently discards the traces its operator asked for is
            worse than one that refuses to start.
    """
    global _tracer, _meter  # noqa: PLW0603 - process-global providers, set once

    if not settings.enabled:
        _tracer, _meter = NoOpTracer(), NoOpMeter()
        return

    trace_api = import_optional("opentelemetry.trace", extra="telemetry", used_by="telemetry")
    metrics_api = import_optional("opentelemetry.metrics", extra="telemetry", used_by="telemetry")
    trace_sdk = import_optional("opentelemetry.sdk.trace", extra="telemetry", used_by="telemetry")
    sampling = import_optional(
        "opentelemetry.sdk.trace.sampling", extra="telemetry", used_by="telemetry"
    )
    export = import_optional(
        "opentelemetry.sdk.trace.export", extra="telemetry", used_by="telemetry"
    )
    metrics_sdk = import_optional(
        "opentelemetry.sdk.metrics", extra="telemetry", used_by="telemetry"
    )
    metrics_export = import_optional(
        "opentelemetry.sdk.metrics.export", extra="telemetry", used_by="telemetry"
    )
    resources = import_optional(
        "opentelemetry.sdk.resources", extra="telemetry", used_by="telemetry"
    )
    span_exporter = import_optional(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        extra="telemetry",
        used_by="telemetry",
    )
    metric_exporter = import_optional(
        "opentelemetry.exporter.otlp.proto.http.metric_exporter",
        extra="telemetry",
        used_by="telemetry",
    )

    resource = resources.Resource.create({"service.name": settings.service_name})

    tracer_provider = trace_sdk.TracerProvider(
        resource=resource,
        # Parent-based, so a sampled trace stays whole: sampling each service
        # independently produces traces with holes in them, which are worse than
        # no trace at all.
        sampler=sampling.ParentBased(sampling.TraceIdRatioBased(settings.sample_ratio)),
    )
    tracer_provider.add_span_processor(
        export.BatchSpanProcessor(
            span_exporter.OTLPSpanExporter(endpoint=f"{settings.endpoint}/v1/traces")
        )
    )
    trace_api.set_tracer_provider(tracer_provider)

    meter_provider = metrics_sdk.MeterProvider(
        resource=resource,
        metric_readers=[
            metrics_export.PeriodicExportingMetricReader(
                metric_exporter.OTLPMetricExporter(endpoint=f"{settings.endpoint}/v1/metrics")
            )
        ],
    )
    metrics_api.set_meter_provider(meter_provider)

    _tracer = OTelTracer(trace_api.get_tracer("sarb"))
    _meter = OTelMeter(metrics_api.get_meter("sarb"))


def reset_telemetry() -> None:
    """Restore the no-op providers. For tests."""
    global _tracer, _meter

    _tracer, _meter = NoOpTracer(), NoOpMeter()


def _attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Coerce a field mapping into what OTel accepts, dropping empties."""
    return {key: _coerce(value) for key, value in attributes.items() if value is not None}


def _coerce(value: Any) -> Any:
    """Coerce one value into an OTel-acceptable attribute type.

    OTel accepts str, bool, int, float, and homogeneous sequences of those.
    Anything else is stringified rather than raising, for the same reason the JSON
    formatter uses ``default=str``: observability code must never be able to break
    the request it is observing.
    """
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple) and all(
        isinstance(item, str | bool | int | float) for item in value
    ):
        return list(value)
    return str(value)
