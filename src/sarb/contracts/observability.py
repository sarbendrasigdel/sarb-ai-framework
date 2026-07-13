"""Ports for logging, tracing, and metrics.

Engines depend on these. They do not depend on ``opentelemetry``, on stdlib
``logging``, or on anything else that could later be replaced — which is enforced
by the architecture tests, not just intended.

Why wrap OpenTelemetry at all
-----------------------------
``opentelemetry-api`` is itself a vendor-neutral API that no-ops without an SDK,
so wrapping it looks like one abstraction too many. It buys two concrete things
that are worth the eighty lines of adapter:

* ``pip install sarb`` pulls in no telemetry stack at all. A core install traces
  nothing and costs nothing.
* An engine can be unit-tested without an SDK, an exporter, or a collector.

The ports are deliberately shaped *like* OpenTelemetry rather than inventing a
new vocabulary, so the adapter is a translation and not a reinterpretation.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

__all__ = ["Logger", "Meter", "Span", "Tracer"]


@runtime_checkable
class Logger(Protocol):
    """Structured logging.

    Fields are keyword arguments, not string interpolation. ``logger.info("llm
    call finished", model="claude-sonnet-5", tokens=812)`` produces a record that
    can be *queried* — "p95 latency where model=claude-sonnet-5" — while
    ``logger.info(f"llm call finished for {model} using {tokens} tokens")``
    produces a record that can only be grepped.

    Values whose key looks like a credential are masked before emission; see
    :mod:`sarb.redaction`.
    """

    def debug(self, event: str, **fields: Any) -> None:
        """Log a message for developers, off in production."""
        ...

    def info(self, event: str, **fields: Any) -> None:
        """Log something a healthy system does."""
        ...

    def warning(self, event: str, **fields: Any) -> None:
        """Log something recoverable that a human may want to know about."""
        ...

    def error(self, event: str, *, exc_info: BaseException | bool = False, **fields: Any) -> None:
        """Log a failure.

        Args:
            event: What failed.
            exc_info: The exception, or ``True`` to use the one being handled.
            **fields: Structured context.
        """
        ...

    def critical(
        self, event: str, *, exc_info: BaseException | bool = False, **fields: Any
    ) -> None:
        """Log a failure that ends the process or the service."""
        ...

    def bind(self, **fields: Any) -> Logger:
        """Return a logger that attaches ``fields`` to every record.

        Composition rather than inheritance: an agent run binds ``run_id`` once
        and every subsequent line carries it, without threading the value through
        a dozen call signatures.
        """
        ...


@runtime_checkable
class Span(Protocol):
    """One unit of work inside a trace."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach a single attribute to this span."""
        ...

    def add_event(self, name: str, **attributes: Any) -> None:
        """Record a point-in-time event within this span."""
        ...

    def record_exception(self, exception: BaseException) -> None:
        """Attach an exception to this span without ending it."""
        ...


@runtime_checkable
class Tracer(Protocol):
    """Creates spans.

    The span context manager marks the span as failed and records the exception
    if the block raises, then re-raises. Callers never have to remember to do
    that, which is the point — the one time it is forgotten is the one trace
    somebody needed.
    """

    def span(self, name: str, **attributes: Any) -> AbstractContextManager[Span]:
        """Open a span for the enclosing block.

        Args:
            name: The operation, e.g. ``"llm.generate"``. Low cardinality —
                identifiers belong in attributes, not in the name.
            **attributes: Initial attributes.

        Returns:
            A context manager yielding the span.

        Example:
            >>> with tracer.span("llm.generate", model="claude-sonnet-5") as span:
            ...     response = await llm.generate(request)
            ...     span.set_attribute("tokens.output", response.usage.output_tokens)
        """
        ...


@runtime_checkable
class Meter(Protocol):
    """Records measurements.

    Instruments are addressed by name rather than created and held, because the
    alternative — every call site owning a counter object — turns a one-line
    measurement into a constructor, a field, and a wiring decision. The adapter
    caches the underlying instruments.
    """

    def counter(
        self, name: str, value: int | float = 1, *, unit: str = "", **attributes: Any
    ) -> None:
        """Add to a monotonically increasing count, e.g. tokens consumed."""
        ...

    def histogram(self, name: str, value: float, *, unit: str = "", **attributes: Any) -> None:
        """Record a distribution sample, e.g. request latency."""
        ...
