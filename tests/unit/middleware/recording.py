"""A tracer and a meter that remember what they were told.

Telemetry that is never asserted on is telemetry that quietly stops working. The
failure is invisible — no exception, no test failure, just an empty dashboard on the
day somebody finally needs it — so the spans and the metrics get tested like any other
output of the system.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecordedSpan:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    exceptions: list[BaseException] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, **attributes: Any) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        self.exceptions.append(exception)


@dataclass
class RecordedMeasurement:
    name: str
    value: float
    attributes: dict[str, Any]


class RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[RecordedSpan]:
        span = RecordedSpan(name=name, attributes=dict(attributes))
        self.spans.append(span)
        yield span


class RecordingMeter:
    def __init__(self) -> None:
        self.counters: list[RecordedMeasurement] = []
        self.histograms: list[RecordedMeasurement] = []

    def counter(
        self, name: str, value: int | float = 1, *, unit: str = "", **attributes: Any
    ) -> None:
        self.counters.append(RecordedMeasurement(name, value, attributes))

    def histogram(self, name: str, value: float, *, unit: str = "", **attributes: Any) -> None:
        self.histograms.append(RecordedMeasurement(name, value, attributes))


class RecordingLogger:
    """A logger that keeps what it was told.

    The real logger's methods are read-only, so a test that wants to assert on a warning
    replaces the whole logger rather than patching a method onto it.
    """

    def __init__(self) -> None:
        self.warnings: list[dict[str, Any]] = []

    def warning(self, message: str, **fields: Any) -> None:
        self.warnings.append({"message": message, **fields})

    def bind(self, **fields: Any) -> RecordingLogger:
        return self


def install(monkeypatch) -> tuple[RecordingTracer, RecordingMeter]:
    """Swap the module-level tracer and meter for ones that keep receipts."""
    tracer, meter = RecordingTracer(), RecordingMeter()
    monkeypatch.setattr("sarb.middleware.guard.get_tracer", lambda: tracer)
    monkeypatch.setattr("sarb.middleware.guard.get_meter", lambda: meter)
    return tracer, meter


def named(measurements: list[RecordedMeasurement], name: str) -> list[RecordedMeasurement]:
    """Every measurement recorded under one instrument."""
    return [m for m in measurements if m.name == name]
