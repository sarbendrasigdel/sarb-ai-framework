# Middleware

Retry, concurrency limits, and telemetry — applied to every component, by the container,
without any adapter or engine knowing they exist.

```yaml
resilience:
  max_attempts: 3
  initial_backoff: 0.5
  max_backoff: 30.0
  max_concurrency: 8
```

That is the whole public interface. There is no `@retry` decorator to remember, no
`with_telemetry(...)` to wrap things in, and no retry loop in your engine.

## The problem this solves

Since Module 5, every adapter has classified its failures with some care. A `429`
becomes a `RateLimitError` carrying `retryable=True` and the provider's own
`Retry-After`. A `401` becomes an `AuthenticationError` that is retryable never. A
`503` is retryable; a `400` is not.

And until this module, **nothing read any of it.** A rate limit went straight to your
caller. The token counts every adapter carefully returned were recorded nowhere. The
observability module built in Module 2 had precisely zero consumers.

This is the consumer.

## What a middleware is

A class that **implements a port by wrapping another implementation of the same port**.

```python
class GuardedLLM:              # IS an LLMProvider. WRAPS an LLMProvider.
    def __init__(self, inner: LLMProvider, guard: Guard) -> None: ...

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return await self._guard.run("chat", lambda: self._inner.generate(request), ...)
```

Because ports are `typing.Protocol`, this works on an adapter the framework has never
seen — including one from a plugin that never imported `sarb`. Composition, not
inheritance, applied to the type system.

They are applied in exactly one place, the DI container, around whatever the registry's
factory returned. So no adapter has to remember them, no engine has to ask for them, and
turning retry off for one component is a line of YAML.

## The order is load-bearing

```
record( retry( limit( AnthropicProvider ) ) )
```

**The limiter is innermost.** If it sat outside the retry, a rate-limited call would
hold its concurrency slot for the whole backoff sleep — so one throttled request would
block a healthy one from even *starting*, and a burst of `429`s would take the component
offline rather than merely slowing it down. Innermost means each attempt takes a slot and
gives it back.

That claim is not decorative. It is asserted in `test_a_backing_off_call_does_not_hold_a_slot`,
and if you move the limiter outward, that test goes red.

**The recording is outermost.** The duration reported is the one the caller actually
waited — retries and backoff included. A histogram that timed only the lucky third
attempt would show a healthy p99 for a system that is, from the outside, unusable.

## Retry

Four rules, each of which exists because the obvious alternative is worse.

**The error decides, not the middleware.** `should_retry` never looks at a status code,
a vendor exception, or a message. It reads `retryable`, a verdict the adapter already
reached. That is what lets it work on a provider nobody has written yet.

**A provider that says *when* is believed.** `Retry-After` beats our computed backoff,
and is *not* capped by `max_backoff` — the ceiling bounds our guessing, and does not get
to overrule an instruction from the only party that knows when the quota resets.

**Backoff has full jitter** — a uniform draw from `[0, min(cap, base × 2ⁿ))`, not that
value exactly. When an indexing job fires forty concurrent embedding batches and all
forty are rate limited in the same instant, an unjittered backoff retries all forty at
precisely `t+1s` — recreating the burst that caused the problem, in lockstep, until the
attempts run out.

**Cancellation is never retried or slept through.** Swallowing it would hang a shutdown
for `max_attempts × max_backoff`, and the process would be killed rather than closed.

A bug is not retried either. A `TypeError` from inside an adapter is a bug, and retrying
it three times only means it takes three times as long to find out.

### Streaming: the subtle one

**You cannot retry a stream after you have emitted from it.** Once a token has reached
the caller it has been rendered, logged, or written down a socket. Retrying would restart
the answer in a user's browser, mid-sentence.

So the guard splits the call in half:

- **Getting to the first event is retried.** A `429` or a dropped connection there has
  produced nothing anyone has seen, so another attempt is invisible — and a rate limit on
  a streaming endpoint should not be fatal when it is fatal nowhere else.
- **Everything after the first event is not.** A failure mid-stream propagates.

Most frameworks pick one half and get the other wrong: either streams are never retried
(so a `429` kills them) or they are retried blindly (so users watch the answer begin
twice).

The concurrency slot is held for the *whole* stream, because a stream occupies the
provider for its whole duration — releasing it after the first token would make
`max_concurrency` a limit on nothing.

## Telemetry

Spans and metrics follow the **OpenTelemetry GenAI semantic conventions**:

| | |
| --- | --- |
| span name | `chat claude-sonnet-5` |
| attributes | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons` |
| metrics | `gen_ai.client.operation.duration` (histogram, seconds), `gen_ai.client.token.usage` (histogram, split by `gen_ai.token.type`) |
| ours | `sarb.component` — the config path. `sarb.retries` — a counter |

The conventions rather than names of our own, for one reason: **every observability
vendor already ships an LLM dashboard keyed on them.** Inventing `sarb.llm.tokens` means
building and maintaining that dashboard ourselves, forever, in every backend the company
ever adopts. They are still evolving, which is exactly the argument for pinning them in
[`middleware/semconv.py`](https://github.com/sarbendra/sarb/blob/main/src/sarb/middleware/semconv.py)
and nowhere else.

`sarb.component` is ours because no convention has it, and it is the attribute anyone
actually wants to group by. An application runs three models and two of them are from the
same vendor; `gen_ai.provider.name` cannot tell them apart, and the config already gave
them names.

`sarb.retries` is the earliest warning there is: retries climb well before failures do.

!!! warning "Prompts do not go into spans"
    Not unless `telemetry.capture_prompts: true`, which is off by default. Prompts
    routinely contain user data, and a tracing backend is not where you want to discover
    that. The same applies to error *messages* on metrics: only the exception class is
    recorded, because a message is unbounded — it carries ids, model names, sometimes
    whole prompts — and an unbounded metric dimension is how a metrics backend falls over.

## Configuration

The application-wide default, and a per-component override:

```yaml
resilience:
  max_attempts: 3
  initial_backoff: 0.5
  max_backoff: 30.0
  max_concurrency: 8

llms:
  default:
    provider: anthropic
    options: { model: claude-sonnet-5 }
  judge:
    provider: openai
    options: { model: gpt-4o }
    resilience:
      max_attempts: 1     # a judge that retries is a judge that bills triple
```

An override **replaces** rather than merges. A partial merge reads as "these two
settings, plus whatever the default happens to be" — which is a sentence nobody can act
on when the default later changes underneath them.

Note that this does *not* violate the rule that [the kernel never learns a vendor's
name](../architecture/overview.md#2-the-kernel-never-learns-a-vendors-name). Resilience
is entirely vendor-neutral — a rate limit means the same thing whoever imposed it — which
is precisely why the kernel is allowed to understand this while `options` stays opaque.

### Retry is on by default

`max_attempts: 3`. This was a deliberate call, and the argument is that **the alternative
is not "no retry"** — it is a retry loop hand-written into every engine by whoever hit a
`429` first, and that loop will not honour `Retry-After`, and it will hammer a provider
that has just asked to be left alone.

`max_attempts: 1` turns it off, per component or globally.

`max_concurrency` defaults to unlimited, which is right for a local model that already
queues internally and wrong for a hosted API. It is a decision, so it is made per
component rather than guessed.

## Why the wrappers are keyed on the *port*

```python
_WRAPPERS = {
    LLMProvider: GuardedLLM,
    EmbeddingProvider: GuardedEmbedder,
}
```

Not on the family name — and the distinction is not academic. A family name is a config
key; a port is a contract. `Family("llms", Greeter)` is a perfectly legal thing for a
plugin or a test to construct, and a table keyed on `"llms"` would cheerfully wrap that
`Greeter` in a `GuardedLLM` and hand back something with no `greet()` on it. **A decorator
decorates a port.** That is the only thing it can safely be selected by.

(This was found by the DI container's own tests, which do exactly that, about ninety
seconds after the first version was written.)

A port with no entry is returned **unwrapped**, deliberately. A plugin that adds a
reranker gets no retry until someone writes `GuardedReranker`. The alternative — a
`__getattr__` proxy that forwards anything — would make `isinstance(x, AnyPortAtAll)`
answer `True` for absolutely anything and defeat the type checker on the way past.

## The N×M problem, and the `Guard`

Three policies times eleven ports is thirty-three classes, and thirty-three chances to get
the backoff subtly different in one of them.

The way out: retry, limiting and instrumentation do not care *what* the call is — only
that it is an async operation that might fail. So the **logic** lives once, in `Guard`,
and only the **shape** is per-port:

```
Guard             one class. limit → retry → record. Knows nothing about LLMs.
GuardedLLM        ~20 lines. Routes generate/stream through the Guard.
GuardedEmbedder   ~12 lines. Routes embed through the Guard.
```

The guard cannot read an `LLMResponse` — it does not know what one is, and it must keep
not knowing, because the same guard wraps embedders and will wrap vector stores. So the
per-port wrapper hands it an `Outcome` describing its own result, and the guard records
what it is given.

## Testing against it

The fake providers can be told to fail a *specific number of times*:

```yaml
llms:
  default:
    provider: fake
    options:
      responses: ["Paris."]
      fail_with: rate_limit
      fail_times: 2        # then it works
```

"Fails forever" cannot express the case that actually matters — rate limited twice, then
fine. Without `fail_times`, a retry policy can only be tested against a provider that
never recovers, which proves the policy gives up and proves nothing about whether it ever
gets an answer.

## API reference

::: sarb.middleware
