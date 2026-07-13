# Observability

Structured logs, traces, and metrics — behind ports, so none of them are a
one-way door.

```python
from sarb.observability import configure_observability, get_logger, get_tracer, log_context

configure_observability(settings)          # once, at startup

logger = get_logger(__name__)
tracer = get_tracer()

with log_context(request_id=req_id):
    with tracer.span("llm.generate", model="claude-sonnet-5") as span:
        response = await llm.generate(request)
        span.set_attribute("tokens.output", response.usage.output_tokens)
        logger.info("generated", output_tokens=response.usage.output_tokens)
```

Tracing and metrics are **no-ops until configured**. A core install pays nothing
for telemetry it does not use, and instrumented code never has to check whether
telemetry happens to be switched on.

## Fields, not f-strings

```python
logger.info("llm call finished", model="claude-sonnet-5", output_tokens=812)
```

```json
{"timestamp": "2026-07-11T10:03:43+00:00", "level": "INFO", "logger": "sarb.agent",
 "event": "llm call finished", "model": "claude-sonnet-5", "output_tokens": 812}
```

That record can be **queried** — p95 latency where `model=claude-sonnet-5`. The
f-string version, `logger.info(f"llm call finished for {model}...")`, produces a
record you can only grep. At three in the morning that is the difference between
an answer and an archaeology project.

## Correlation: `log_context`

One request fans out through a retrieval engine, three tool calls and two model
calls, and produces forty log lines across a dozen modules. Every one of them
needs the `request_id`, or none of them can be reassembled into the story of what
happened to one user.

Threading a correlation id through forty call signatures is not a design; it is a
tax that gets skipped the moment someone is in a hurry. So it lives in a
`ContextVar`, which asyncio propagates into every task spawned inside the block:

```python
with log_context(request_id=req_id, user_id=user.id):
    await handle(request)          # every record inside carries both
```

Context is **additive and nested**, and unwinds exactly — including when the block
raises:

```python
with log_context(request_id="req-1"):
    with log_context(agent="researcher"):
        logger.info("step")     # request_id AND agent
    logger.info("done")         # request_id only
```

Precedence, lowest to highest: **context → bound → call-site.** The most specific
thing wins.

## Errors carry their own fields

Every framework error has a structured `context`. Logging one lifts that context
onto the record automatically:

```python
except RateLimitError as exc:
    logger.error("llm call failed", exc_info=exc)
```

```json
{"level": "ERROR", "event": "llm call failed", "error_type": "RateLimitError",
 "provider": "anthropic", "retry_after": 30.0, "exception": "Traceback..."}
```

`provider=anthropic` is a field you can filter on, not a fragment of a message
string you have to regex out — and no call site had to remember to unpack it.

## Redaction

Values whose key names a credential are masked before emission. So are all
`SecretStr` values, whatever key they sit under.

```python
logger.info("configured", model="claude-sonnet-5", api_key="sk-ant-live")
# -> {"model": "claude-sonnet-5", "api_key": "***"}
```

### The matching rule, and why it is not a substring match

The key is normalised (`apiKey`, `API_KEY`, `x-api-key` → `api_key`) and then
matched **exactly, or on a trailing `_hint` segment**.

The obvious implementation — substring matching — is wrong in this domain, and
dangerously so:

| Key | Substring rule | This rule |
| --- | --- | --- |
| `api_key` | masked ✅ | masked ✅ |
| `access_token` | masked ✅ | masked ✅ |
| `output_tokens` | **masked** ❌ | visible ✅ |
| `token_count` | **masked** ❌ | visible ✅ |

`"token"` is a substring of `"tokens"`. A naive substring rule silently redacts
every token count in every log line — destroying the single most important metric
an LLM framework emits, and doing it quietly.

Where the two goals genuinely conflict the bias is still toward over-masking: a
needlessly masked field costs a developer a moment of confusion, while a missed
one puts a live credential in a log aggregator that a dozen people can read and
that retains it for a year.

Add your own:

```yaml
logging:
  redact_keys: [api_key, authorization, password, secret, token, ssn]
```

## Tracing

```python
with tracer.span("llm.generate", model="claude-sonnet-5") as span:
    response = await llm.generate(request)
    span.set_attribute("tokens.output", response.usage.output_tokens)
```

Span names are **low cardinality** — `llm.generate`, not `llm.generate.user-42`.
Identifiers belong in attributes.

A block that raises marks the span as errored and records the exception, then
re-raises. Callers never have to remember to do that, which is the point: the one
span somebody forgets is always the one that mattered.

### Logs and traces are correlated automatically

When OpenTelemetry is active, opening a span binds `trace_id` and `span_id` into
the logging context. Every log line inside the span carries them, so *"show me the
logs for this trace"* is a query rather than an exercise in timestamp arithmetic.

## Metrics

```python
meter = get_meter()
meter.counter("llm.tokens", response.usage.output_tokens, unit="token", model="claude-sonnet-5")
meter.histogram("llm.latency", elapsed, unit="s", model="claude-sonnet-5")
```

Instruments are addressed by name rather than constructed and held. The
alternative turns a one-line measurement into a constructor, a field, and a
wiring decision — so it does not get written. The adapter caches the underlying
OTel instruments.

## Turning telemetry on

```yaml
telemetry:
  enabled: true
  service_name: sarb
  endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT}
  sample_ratio: 0.1
  capture_prompts: false # prompts contain user data
```

```bash
uv add "sarb[telemetry]"
```

Enabling telemetry **without** the extra installed is a hard error naming the
install command. A process that silently discards the traces its operator asked
for is worse than one that refuses to start.

Sampling is **parent-based**: a trace that is sampled stays whole. Sampling each
service independently produces traces with holes in them, which are worse than no
trace at all.

!!! warning "`capture_prompts` is off by default"
    Prompts routinely contain user data. Attaching them to spans ships that data
    to your tracing backend, which is very likely a different trust boundary from
    your database. Turn it on deliberately, per environment, or not at all.

## Design decisions

### Why a `Logger` port instead of just using structlog

structlog is a good library. But logging is the one dependency *every* module
touches, so it is the one worth keeping reversible. Building the default on
stdlib `logging` means the core has no logging dependency at all, and a team that
prefers structlog — or Loguru, or a house logger — implements six methods and
binds it, with no fork.

### Why wrap OpenTelemetry

`opentelemetry-api` is itself a vendor-neutral API that no-ops without an SDK, so
wrapping it looks like one abstraction too many. It buys exactly two things, and
they are worth the eighty lines of adapter:

- `pip install sarb` pulls in no telemetry stack at all.
- An engine can be unit-tested without an SDK, an exporter, or a collector.

The ports are shaped *like* OpenTelemetry rather than inventing a new vocabulary,
so the adapter is a translation and not a reinterpretation.

### Why the providers are process-global

This deliberately breaks from how [configuration](configuration.md#immutability-and-the-absence-of-a-global)
is handled, where a global is refused.

The difference: config *drives behaviour* and must be injectable and swappable per
test. A logger is an **output sink**. Threading one through every constructor in
the framework is ceremony that buys nothing, and both stdlib `logging` and
OpenTelemetry converged on global providers for the same reason.

The cost is that logs would be awkward to assert on — so the framework provides
the seam rather than leaving it there. See below.

### Logging must never break the request it describes

A field holding a UUID, a `datetime`, or somebody's dataclass must not be able to
raise inside a logging call and take down the request it was trying to describe.
JSON serialisation falls back to `str`, and OTel attributes are coerced rather
than rejected. Observability code is allowed to render a value badly. It is never
allowed to fail.

## Testing

```python
from sarb.observability.testing import capture_logs

with capture_logs() as records:
    await agent.run("hello")

assert records.first("agent.finished")["iterations"] == 3
assert records.events() == ["agent.started", "llm.called", "agent.finished"]
```

Redaction still applies inside `capture_logs`, so a test can assert that a secret
*was* masked — the helper is not a hole in the guarantee.

Previously-installed handlers are suspended and restored, so capturing logs in one
test does not silently disable logging for the ones that run after it.

## API reference

::: sarb.observability

::: sarb.contracts.observability

::: sarb.redaction
