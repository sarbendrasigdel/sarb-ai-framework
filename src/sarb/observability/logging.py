"""Structured logging on top of the standard library.

Why not structlog
-----------------
It is a good library. But logging is the one dependency that every single module
touches, and the framework defines a :class:`~sarb.contracts.observability.Logger`
port precisely so that choice stays reversible. Building the default on stdlib
``logging`` means the core has no logging dependency at all, and a team that
wants structlog — or Loguru, or their own house logger — implements six methods
and binds it, without a fork.

The implementation is thin by design: field merging, redaction, and a formatter.

Why fields, not f-strings
-------------------------
``logger.info("llm call finished", model="claude-sonnet-5", tokens=812)`` produces
a record you can *query*: p95 latency where ``model=claude-sonnet-5``. The f-string
version produces a record you can only grep. At three in the morning that is the
whole difference between an answer and an archaeology project.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, ClassVar, Final

from sarb.config import LoggingSettings
from sarb.contracts.observability import Logger
from sarb.errors import SarbError
from sarb.observability.context import current_context
from sarb.redaction import DEFAULT_SECRET_HINTS, redact

__all__ = ["StdlibLogger", "configure_logging", "get_logger"]

#: Where structured fields ride on a LogRecord. One key, not many, so that a field
#: called `name` or `module` cannot collide with a LogRecord attribute and corrupt
#: the record — a failure mode that is maddening to diagnose because the logging
#: call looks completely ordinary.
FIELDS_ATTR: Final = "sarb_fields"

#: LogRecord attributes that are never treated as user fields.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    FIELDS_ATTR,
    "message",
    "asctime",
    "taskName",
}

#: Marks handlers this module installed, so reconfiguring replaces them instead of
#: stacking a second copy and emitting everything twice.
_MANAGED = "_sarb_managed"


class StdlibLogger:
    """The default :class:`~sarb.contracts.observability.Logger`.

    Args:
        logger: The underlying stdlib logger.
        fields: Fields bound to this instance, attached to every record.
    """

    __slots__ = ("_fields", "_logger")

    def __init__(self, logger: logging.Logger, fields: dict[str, Any] | None = None) -> None:
        self._logger = logger
        self._fields = fields or {}

    def bind(self, **fields: Any) -> Logger:
        """Return a new logger carrying these fields in addition to its own.

        The receiver is not modified: binding is composition, so an agent can hand
        a bound logger to a sub-task without the sub-task's fields leaking back.
        """
        return StdlibLogger(self._logger, {**self._fields, **fields})

    def debug(self, event: str, **fields: Any) -> None:
        """Log at DEBUG."""
        self._emit(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: Any) -> None:
        """Log at INFO."""
        self._emit(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        """Log at WARNING."""
        self._emit(logging.WARNING, event, fields)

    def error(self, event: str, *, exc_info: BaseException | bool = False, **fields: Any) -> None:
        """Log at ERROR, optionally with an exception and its context."""
        self._emit(logging.ERROR, event, self._with_error(exc_info, fields), exc_info=exc_info)

    def critical(
        self, event: str, *, exc_info: BaseException | bool = False, **fields: Any
    ) -> None:
        """Log at CRITICAL, optionally with an exception and its context."""
        self._emit(logging.CRITICAL, event, self._with_error(exc_info, fields), exc_info=exc_info)

    @staticmethod
    def _with_error(exc_info: BaseException | bool, fields: dict[str, Any]) -> dict[str, Any]:
        """Lift a SarbError's structured context onto the record.

        Every framework error carries a ``context`` dict. Merging it here is what
        makes a failure queryable — ``provider=anthropic`` becomes a field rather
        than a fragment of a message string — without every call site having to
        remember to unpack it.
        """
        if isinstance(exc_info, SarbError):
            return {**exc_info.context, "error_type": type(exc_info).__name__, **fields}
        if isinstance(exc_info, BaseException):
            return {"error_type": type(exc_info).__name__, **fields}
        return fields

    def _emit(
        self,
        level: int,
        event: str,
        fields: dict[str, Any],
        *,
        exc_info: BaseException | bool = False,
    ) -> None:
        """Merge ambient, bound, and call-site fields, then hand off to stdlib."""
        if not self._logger.isEnabledFor(level):
            # Skip the merge entirely on a disabled level. DEBUG logging in a hot
            # retrieval loop should cost approximately nothing in production.
            return

        merged = {**current_context(), **self._fields, **fields}
        self._logger.log(
            level,
            event,
            exc_info=exc_info if isinstance(exc_info, BaseException | bool) else None,
            extra={FIELDS_ATTR: merged},
        )


def get_logger(name: str, **fields: Any) -> Logger:
    """Return a logger for a module.

    Args:
        name: Usually ``__name__``.
        **fields: Fields bound to every record from this logger.

    Returns:
        A :class:`~sarb.contracts.observability.Logger`.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("retrieved", chunks=8, latency_ms=41)
    """
    return StdlibLogger(logging.getLogger(name), dict(fields))


class _StructuredFormatter(logging.Formatter):
    """Shared field extraction: pull, redact, and never crash the caller."""

    def __init__(self, redact_keys: frozenset[str]) -> None:
        super().__init__()
        self._hints = redact_keys

    def _fields(self, record: logging.LogRecord) -> dict[str, Any]:
        """Structured fields for a record, redacted.

        Records from third-party libraries have no ``sarb_fields``; their custom
        ``extra`` attributes are recovered by diffing against the reserved set, so
        a library that logs ``extra={"status": 500}`` still produces a field.
        """
        fields: dict[str, Any] = dict(getattr(record, FIELDS_ATTR, {}))
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in fields:
                fields[key] = value
        return dict(redact(fields, hints=self._hints))


class JsonFormatter(_StructuredFormatter):
    """One JSON object per line. The production format.

    ``default=str`` on the dump is load-bearing: a field holding a UUID, a
    ``datetime``, or somebody's dataclass must not be able to raise inside the
    logging call and take down the request it was trying to describe. Logging is
    allowed to render a value badly; it is never allowed to fail.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render the record as a single-line JSON object."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **self._fields(record),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(_StructuredFormatter):
    """Human-readable, for a terminal. Never the production format."""

    _COLOURS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    _RESET = "\033[0m"

    def __init__(self, redact_keys: frozenset[str], *, colour: bool = True) -> None:
        super().__init__(redact_keys)
        self._colour = colour

    def format(self, record: logging.LogRecord) -> str:
        """Render the record as an aligned, optionally coloured line."""
        stamp = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")
        level = record.levelname
        if self._colour:
            level = f"{self._COLOURS.get(level, '')}{level:<8}{self._RESET}"
        else:
            level = f"{level:<8}"

        fields = self._fields(record)
        rendered = " ".join(f"{k}={v!r}" for k, v in sorted(fields.items()))

        line = f"{stamp} {level} {record.name:<28} {record.getMessage()}"
        if rendered:
            line = f"{line}  {rendered}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(settings: LoggingSettings, *, stream: Any = None) -> None:
    """Install the framework's logging configuration on the root logger.

    Called once, from the composition root. Idempotent: calling it again replaces
    the previous handler rather than adding a second one, so a process that
    reconfigures does not start emitting every line twice.

    Args:
        settings: The ``logging`` section of the application settings.
        stream: Where to write. Defaults to stderr, which is what a container
            runtime collects and what keeps logs out of stdout when stdout is the
            program's actual output (a CLI printing results, say).

    Example:
        >>> configure_logging(settings.logging)
        >>> get_logger(__name__).info("ready", port=8000)
    """
    hints = DEFAULT_SECRET_HINTS | {k.lower() for k in settings.redact_keys}

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        JsonFormatter(hints)
        if settings.format == "json"
        else ConsoleFormatter(hints, colour=_supports_colour(stream))
    )
    setattr(handler, _MANAGED, True)

    root = logging.getLogger()
    for existing in [h for h in root.handlers if getattr(h, _MANAGED, False)]:
        root.removeHandler(existing)
        existing.close()

    root.addHandler(handler)
    root.setLevel(settings.level.value)

    # Warnings are things somebody wants to know about; the default is to print
    # them to stderr unstructured, where a log aggregator cannot see them.
    logging.captureWarnings(capture=True)


def _supports_colour(stream: Any) -> bool:
    """Whether ANSI colour is safe to emit. Escape codes in a log file are noise."""
    target = stream or sys.stderr
    return bool(getattr(target, "isatty", lambda: False)())
