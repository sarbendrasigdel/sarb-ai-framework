"""Observability: structured logs, traces, and metrics.

    from sarb.observability import configure_observability, get_logger, get_tracer

    configure_observability(settings)          # once, at startup

    logger = get_logger(__name__)
    tracer = get_tracer()

    with log_context(request_id=req_id):       # every record below carries it
        with tracer.span("llm.generate", model="claude-sonnet-5") as span:
            response = await llm.generate(request)
            span.set_attribute("tokens.output", response.usage.output_tokens)
            logger.info("generated", tokens=response.usage.output_tokens)

Tracing and metrics are **no-ops until configured**, so a core install pays
nothing for telemetry it does not use. Logging is always on.

Engines depend on the ports in :mod:`sarb.contracts.observability`, never on
``opentelemetry`` — enforced by the architecture tests.
"""

from sarb.contracts.observability import Logger, Meter, Span, Tracer
from sarb.observability.context import current_context, log_context
from sarb.observability.logging import configure_logging, get_logger
from sarb.observability.setup import configure_observability
from sarb.observability.telemetry import configure_telemetry, get_meter, get_tracer

__all__ = [
    "Logger",
    "Meter",
    "Span",
    "Tracer",
    "configure_logging",
    "configure_observability",
    "configure_telemetry",
    "current_context",
    "get_logger",
    "get_meter",
    "get_tracer",
    "log_context",
]
