"""Helpers for asserting on what the framework logged.

Logging providers are process-global, which is the right call for ergonomics but
would make logs awkward to assert on if the framework left it there. So it does
not: :func:`capture_logs` gives a test a list of records to make assertions
against, and restores whatever was configured before.

    with capture_logs() as records:
        await agent.run("hello")

    assert records.first("agent.finished")["iterations"] == 3
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sarb.observability.logging import FIELDS_ATTR
from sarb.redaction import DEFAULT_SECRET_HINTS, redact

__all__ = ["CapturedLogs", "capture_logs"]


class CapturedLogs(list[dict[str, Any]]):
    """The records emitted inside a :func:`capture_logs` block.

    Each entry is a flat dict: ``event``, ``level``, ``logger``, plus every
    structured field, redacted exactly as it would be on the way to a real sink —
    so a test can assert that a secret *was* masked.
    """

    def events(self) -> list[str]:
        """Every event name, in order. Useful for asserting a sequence."""
        return [record["event"] for record in self]

    def find(self, event: str) -> list[dict[str, Any]]:
        """Every record with this event name."""
        return [record for record in self if record["event"] == event]

    def first(self, event: str) -> dict[str, Any]:
        """The first record with this event name.

        Raises:
            AssertionError: If no record matches. The message lists what *was*
                logged, which is almost always enough to see why.
        """
        matches = self.find(event)
        if not matches:
            raise AssertionError(f"No log record for event {event!r}. Got: {self.events()}")
        return matches[0]

    def has(self, event: str) -> bool:
        """Whether any record has this event name."""
        return any(record["event"] == event for record in self)


class _CaptureHandler(logging.Handler):
    """Collects records as plain dicts, applying the real redaction rules."""

    def __init__(self, records: CapturedLogs) -> None:
        super().__init__()
        self._records = records

    def emit(self, record: logging.LogRecord) -> None:
        """Flatten one record into the capture list."""
        fields = dict(getattr(record, FIELDS_ATTR, {}))
        self._records.append(
            {
                "event": record.getMessage(),
                "level": record.levelname,
                "logger": record.name,
                **dict(redact(fields, hints=DEFAULT_SECRET_HINTS)),
            }
        )


@contextmanager
def capture_logs(level: int = logging.DEBUG) -> Iterator[CapturedLogs]:
    """Capture everything logged inside the block.

    Existing handlers are suspended for the duration and restored afterwards, so a
    test that captures logs does not silently disable logging for the tests that
    run after it.

    Args:
        level: Minimum level to capture.

    Yields:
        The records, which fill up as the block runs.

    Example:
        >>> with capture_logs() as records:
        ...     get_logger("x").info("ready", port=8000)
        >>> records.first("ready")["port"]
        8000
    """
    records = CapturedLogs()
    handler = _CaptureHandler(records)

    root = logging.getLogger()
    previous_handlers = root.handlers[:]
    previous_level = root.level

    root.handlers = [handler]
    root.setLevel(level)
    try:
        yield records
    finally:
        root.handlers = previous_handlers
        root.setLevel(previous_level)
