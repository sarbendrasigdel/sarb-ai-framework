"""The names of things, pinned in one place.

Telemetry attribute and metric names are an interface — with dashboards, alerts and
queries on the other side of it. Scattering the string ``"gen_ai.usage.input_tokens"``
across four adapters means a typo in one of them is a metric that silently does not
exist, and nobody notices until the graph they needed is empty.

So the names live here, once.

Why the OpenTelemetry GenAI conventions rather than our own
-----------------------------------------------------------
The obvious alternative is ``sarb.llm.tokens``, which reads better and is ours. The
problem is that every observability vendor already ships an LLM dashboard keyed on the
GenAI attribute names — and inventing our own means building and maintaining that
dashboard ourselves, forever, in every backend the company ever adopts.

The conventions are still evolving. That is precisely the argument for pinning them in
one module: when they change, this file changes, and nothing else does.
"""

from __future__ import annotations

__all__ = [
    "COMPONENT",
    "COST",
    "DURATION",
    "ERROR_TYPE",
    "FINISH_REASONS",
    "INPUT_TOKENS",
    "OPERATION",
    "OUTPUT_TOKENS",
    "PROVIDER",
    "REQUEST_MODEL",
    "RESPONSE_MODEL",
    "RETRIES",
    "TARGET",
    "TOKENS",
    "TOKEN_TYPE",
]

# -- span and metric attributes ---------------------------------------------- #

OPERATION = "gen_ai.operation.name"
"""``chat`` or ``embeddings``. Low cardinality, by design."""

PROVIDER = "gen_ai.provider.name"
"""The registry key: ``anthropic``, ``openai``, ``fastembed``."""

REQUEST_MODEL = "gen_ai.request.model"
"""What we asked for."""

RESPONSE_MODEL = "gen_ai.response.model"
"""What actually served it. Providers alias and silently upgrade, so the two differ."""

INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
FINISH_REASONS = "gen_ai.response.finish_reasons"

TOKEN_TYPE = "gen_ai.token.type"
"""Splits the token histogram. One of:

``input``
    Everything sent, cache reads and writes included. Summing every token type would
    therefore double-count, which is deliberate: this dimension exists so a dashboard can
    *filter*, and a dashboard that wants the total already has ``input``.
``output``
    Everything generated, reasoning included.
``cached_input``, ``cache_write_input``
    The parts of ``input`` that were billed at a cache rate rather than the full one.
    Without these, a cost estimate cannot see the discount it is getting.
``reasoning``
    The part of ``output`` spent thinking. Billed as output, and often most of it.
"""

ERROR_TYPE = "error.type"
"""The exception class, e.g. ``RateLimitError``. Not the message: that is unbounded."""

TARGET = "sarb.target"
"""What the call was aimed at: a model for an LLM, a collection for a vector store.

The span name is built from it, so it must stay low cardinality. It exists because
`gen_ai.request.model` is the wrong home for a collection name — stamping "documents"
into a field every observability vendor reads as a model would put fictional models on
everybody's GenAI dashboard.
"""

COMPONENT = "sarb.component"
"""``llms.default``, ``llms.judge``, ``embedders.default``.

Ours, because no convention has it, and because it is the attribute anyone actually
wants to group by. An application runs three models and two of them are the same
vendor; ``gen_ai.provider.name`` cannot tell them apart, and the config already gave
them names.
"""

# -- instruments -------------------------------------------------------------- #

DURATION = "gen_ai.client.operation.duration"
"""Histogram, seconds. Wall clock as the *caller* experienced it — retries included.

A histogram that timed only the attempt that happened to succeed would report a
healthy p99 for a system that is, from the outside, timing out.
"""

TOKENS = "gen_ai.client.token.usage"
"""Histogram, tokens. Split by :data:`TOKEN_TYPE`."""

RETRIES = "sarb.retries"
"""Counter. Ours, because nothing else counts this and it is the earliest warning there
is: retries climb well before failures do."""

COST = "sarb.cost.usd"
"""Counter, US dollars. Ours, because the GenAI conventions have no cost metric — cost is
not something a provider reports, it is something you compute from tokens and a price
list, and OpenTelemetry has no opinion on where the price list lives.

A counter rather than a histogram, because the question is *"what did we spend"*, and
that is a sum. The distribution of per-call cost is derivable from the token histograms
by anyone who wants it.

Zero is not recorded, and neither is an unpriced model — see :mod:`sarb.middleware.cost`.
A dashboard that reads this metric must be able to trust that a number here is a number
somebody will be invoiced for."""
