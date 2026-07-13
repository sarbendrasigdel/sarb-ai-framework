"""One call, guarded: limited, retried, and recorded.

This is the whole of the middleware's logic. Everything else in this package is a
handful of lines of delegation that route a port's methods through here.

The split matters. Retry, concurrency-limiting and instrumentation do not care *what*
the call is — only that it is an async operation that might fail. So the logic lives
once, in this class, and the *shape* — ``generate``, ``embed``, ``search`` — is the
only thing that is written per port. The alternative is three policies times eleven
ports, which is thirty-three classes and thirty-three chances to get the backoff
subtly different in one of them.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import anyio

from sarb.config import ModelPrice, ResilienceSettings
from sarb.contracts import Span, Usage
from sarb.middleware import semconv
from sarb.middleware.cost import cost_of
from sarb.middleware.policy import backoff_for, should_retry
from sarb.observability import get_logger, get_meter, get_tracer

__all__ = ["Guard", "Outcome"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a completed call turned out to be worth recording.

    The guard cannot read an :class:`~sarb.contracts.LLMResponse` — it does not know
    what one is, and it must keep not knowing, because the same guard wraps embedders
    and vector stores. So the per-port wrapper describes its own result, and the guard
    records what it is handed.

    Attributes:
        model: The model that actually served the call, when the provider said. Cost is
            priced against *this*, not against what was asked for, because an alias that
            quietly resolves to a dearer model is a thing that happens and a thing the
            invoice will reflect.
        usage: Normalised token counts, or ``None`` for a port that has none — a vector
            store search consumes no tokens and must not pretend to.
        finish_reason: Why generation stopped.
    """

    model: str | None = None
    usage: Usage | None = None
    finish_reason: str | None = None


class Guard:
    """Wraps one call to one component: limits it, retries it, records it.

    Args:
        component: The config path, e.g. ``"llms.judge"``. Every log line, span and
            metric carries it, which is what makes "why is the judge slow" a query.
        provider: The registry key, e.g. ``"anthropic"``.
        policy: Retry and concurrency settings for this component.

    The three concerns compose in a fixed order, and the order is load-bearing:

    ``record( retry( limit( the actual call ) ) )``

    **The limiter is innermost.** If it sat outside the retry, a call that was rate
    limited would hold its concurrency slot for the whole backoff sleep — so one
    throttled request would block a healthy one from even starting, and a burst of rate
    limits would take the component offline rather than merely slowing it down.
    Innermost means each attempt takes a slot and gives it back, so a call that is
    waiting is not a call that is occupying anything.

    **The recording is outermost.** The duration reported is the one the caller
    actually waited, retries and backoff included. A histogram that timed only the
    lucky third attempt would show a healthy p99 for a system that is, from the
    outside, unusable.
    """

    def __init__(
        self,
        *,
        component: str,
        provider: str,
        policy: ResilienceSettings,
        pricing: dict[str, dict[str, ModelPrice]] | None = None,
    ) -> None:
        self._component = component
        self._provider = provider
        self._policy = policy
        self._pricing = pricing or {}
        self._limiter = (
            anyio.CapacityLimiter(policy.max_concurrency) if policy.max_concurrency else None
        )
        self._log = _log.bind(component=component, provider=provider)
        #: Models we have already complained about having no price for. Bounded by the
        #: number of models one component talks to, which is one, so this is not a leak —
        #: and warning once beats warning on every call, which is how a useful warning
        #: gets filtered out of the logs by everybody.
        self._unpriced: set[str] = set()

    # -- the two entry points ------------------------------------------------- #

    async def run[T](
        self,
        operation: str,
        call: Callable[[], Coroutine[Any, Any, T]],
        *,
        target: str,
        attributes: Mapping[str, Any] | None = None,
        outcome: Callable[[T], Outcome] | None = None,
    ) -> T:
        """Run a unary call — ``generate``, ``embed`` — under the full policy.

        Args:
            operation: ``"chat"`` or ``"embeddings"``. Names the span, and is a
                low-cardinality metric dimension.
            call: A *factory* for the coroutine, not the coroutine. It is invoked
                again on each attempt — a coroutine object cannot be awaited twice,
                so accepting one would make retry impossible.
            target: What the call was aimed at — a model, or a collection. Names the
                span, so it must be low cardinality.
            attributes: Extra span and metric attributes the *wrapper* knows to be
                right. The guard does not know whether it is guarding a language model
                or a database, and it must not guess: writing a collection name into
                ``gen_ai.request.model`` would put fictional models on every GenAI
                dashboard in the company.
            outcome: Reads the result for the things worth recording.

        Returns:
            Whatever ``call`` returned.

        Raises:
            SarbError: The last failure, once the attempts run out — or immediately, if
                the error was never retryable in the first place.
        """
        started = time.perf_counter()

        with self._span(operation, target, attributes) as span:
            attempt = 0
            while True:
                attempt += 1
                try:
                    async with self._slot():
                        result = await call()
                except BaseException as exc:
                    if await self._retrying(exc, attempt=attempt, operation=operation):
                        continue
                    self._record_failure(span, exc, operation, target, attributes, started)
                    raise

                self._record_success(
                    span,
                    outcome(result) if outcome else Outcome(),
                    operation,
                    target,
                    attributes,
                    started,
                )
                return result

    async def stream[T](
        self,
        operation: str,
        call: Callable[[], AsyncIterator[T]],
        *,
        target: str,
        attributes: Mapping[str, Any] | None = None,
        outcome: Callable[[T], Outcome | None] | None = None,
    ) -> AsyncIterator[T]:
        """Run a streaming call under a policy that stops retrying once it commits.

        A stream cannot be retried the way a unary call can, and pretending otherwise
        is how users end up watching an answer begin twice. So the guard splits the
        call in half:

        * **Getting to the first event is retried.** A ``429`` or a dropped connection
          here has produced nothing anyone has seen, so another attempt is invisible.
        * **Everything after the first event is not.** Once a token has been yielded to
          the caller it has been rendered, logged, or sent down a socket. It cannot be
          taken back, so a failure mid-stream propagates.

        The concurrency slot is held for the **whole** stream, because a stream occupies
        the provider for its whole duration — releasing it after the first token would
        make ``max_concurrency`` a limit on nothing. It is released and re-acquired
        across retries, so a backing-off stream is not squatting on a slot.

        Args:
            operation: ``"chat"``. Names the span, and is a metric dimension.
            call: A *factory* for the iterator, not the iterator. Invoked again on each
                attempt, because a half-consumed stream cannot be restarted.
            target: What the call was aimed at. Names the span, so: low cardinality.
            attributes: Extra span and metric attributes the wrapper knows to be right.
            outcome: Called on every event, and expected to return ``None`` for all but
                the terminal one — the last non-``None`` answer is what gets recorded.
                This is how a stream reports its tokens, and it is not optional in
                practice: without it a streamed call records no usage and no cost at all,
                and since chat interfaces stream, that is most of the bill missing from
                every dashboard in the company.
        """
        started = time.perf_counter()

        with self._span(operation, target, attributes) as span:
            attempt = 0
            while True:
                attempt += 1
                events = call()  # a fresh iterator per attempt
                await self._acquire()

                try:
                    first = await anext(events)
                except StopAsyncIteration:
                    # A stream that ends before it starts. Legal, if strange.
                    self._release()
                    self._record_success(span, Outcome(), operation, target, attributes, started)
                    return
                except BaseException as exc:
                    self._release()
                    if await self._retrying(exc, attempt=attempt, operation=operation):
                        continue
                    self._record_failure(span, exc, operation, target, attributes, started)
                    raise

                break

            final = Outcome()

            def observe(event: T) -> None:
                # The usage arrives on the *last* event, so every event is offered to the
                # hook and the last non-None answer wins. Waiting to inspect only the
                # final event would mean holding events back to know which one that is,
                # and delaying the token a user is watching for is not a trade worth
                # making to save an if-statement.
                nonlocal final
                if outcome is not None and (seen := outcome(event)) is not None:
                    final = seen

            # Committed. The slot stays held until the stream is done with.
            try:
                observe(first)
                yield first

                async for event in events:
                    observe(event)
                    yield event
            except BaseException as exc:
                self._record_failure(span, exc, operation, target, attributes, started)
                raise
            finally:
                self._release()

            self._record_success(span, final, operation, target, attributes, started)

    # -- retrying ------------------------------------------------------------- #

    async def _retrying(self, exc: BaseException, *, attempt: int, operation: str) -> bool:
        """Decide whether to try again, and if so, wait first.

        Returns ``True`` having already slept, so the caller's loop just continues.
        """
        # Cancellation is not a failure and must never be retried or slept through.
        # Swallowing it here would hang a shutdown for max_attempts x max_backoff,
        # and the process would be killed rather than closed.
        if isinstance(exc, anyio.get_cancelled_exc_class()):
            raise exc

        if not should_retry(exc, attempt=attempt, policy=self._policy):
            return False

        delay = backoff_for(exc, attempt=attempt, policy=self._policy)

        get_meter().counter(
            semconv.RETRIES,
            1,
            **{
                semconv.COMPONENT: self._component,
                semconv.PROVIDER: self._provider,
                semconv.OPERATION: operation,
                semconv.ERROR_TYPE: type(exc).__name__,
            },
        )
        self._log.warning(
            "retrying after a retryable failure",
            operation=operation,
            attempt=attempt,
            of=self._policy.max_attempts,
            sleeping=round(delay, 3),
            error=type(exc).__name__,
            detail=str(exc),
        )

        await anyio.sleep(delay)
        return True

    # -- limiting ------------------------------------------------------------- #

    def _slot(self) -> AbstractAsyncContextManager[Any]:
        """The concurrency slot for one attempt, or nothing if unlimited."""
        return self._limiter or nullcontext()

    async def _acquire(self) -> None:
        """Take a slot, waiting if they are all busy."""
        if self._limiter:
            await self._limiter.acquire()

    def _release(self) -> None:
        """Give a slot back."""
        if self._limiter:
            self._limiter.release()

    # -- recording ------------------------------------------------------------ #

    def _attributes(
        self, operation: str, target: str, extra: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """What every span and every metric for this call carries.

        The guard contributes only what it actually knows: which component, which
        provider, which operation, and what it was aimed at. Anything domain-specific —
        that the target happens to be a *model*, for instance — is the wrapper's to add,
        because the wrapper is the only thing here that knows what it is wrapping.
        """
        return {
            semconv.OPERATION: operation,
            semconv.PROVIDER: self._provider,
            semconv.COMPONENT: self._component,
            semconv.TARGET: target,
            **(extra or {}),
        }

    def _span(
        self, operation: str, target: str, extra: Mapping[str, Any] | None
    ) -> AbstractContextManager[Span]:
        """Open the span for a whole logical call, retries and all."""
        # `chat claude-sonnet-5`, `search documents`. Both halves are low cardinality,
        # which is what a span name has to be.
        return get_tracer().span(
            f"{operation} {target}", **self._attributes(operation, target, extra)
        )

    def _record_success(
        self,
        span: Span,
        outcome: Outcome,
        operation: str,
        target: str,
        extra: Mapping[str, Any] | None,
        started: float,
    ) -> None:
        """Attach what happened to the span, and to the instruments."""
        attributes = self._attributes(operation, target, extra)
        meter = get_meter()

        if outcome.model:
            span.set_attribute(semconv.RESPONSE_MODEL, outcome.model)
        if outcome.finish_reason:
            span.set_attribute(semconv.FINISH_REASONS, [outcome.finish_reason])

        if outcome.usage is not None:
            self._record_tokens(span, outcome.usage, attributes)
            self._record_cost(span, outcome, attributes)

        meter.histogram(semconv.DURATION, time.perf_counter() - started, unit="s", **attributes)

    def _record_tokens(self, span: Span, usage: Usage, attributes: Mapping[str, Any]) -> None:
        """Split the token histogram five ways.

        The two totals plus their three billed-differently parts. The parts overlap the
        totals on purpose — ``cached_input`` is *inside* ``input`` — because the
        alternative is a dimension you have to add up to get the number you wanted, and
        anyone summing across ``gen_ai.token.type`` to find "how many tokens" would then
        get an answer larger than the truth. A dashboard filters. It should not have to
        do arithmetic to avoid being lied to.

        A zero is never recorded. A local embedder consumes no billable tokens at all,
        and a flat line of zeroes on a graph is indistinguishable from a provider that
        has stopped reporting.
        """
        meter = get_meter()

        for kind, tokens in (
            ("input", usage.input_tokens),
            ("output", usage.output_tokens),
            ("cached_input", usage.cached_input_tokens),
            ("cache_write_input", usage.cache_write_input_tokens),
            ("reasoning", usage.reasoning_tokens),
        ):
            if tokens:
                meter.histogram(
                    semconv.TOKENS,
                    tokens,
                    unit="{token}",
                    **{**attributes, semconv.TOKEN_TYPE: kind},
                )

        # Only the two totals go on the span. The GenAI conventions have attributes for
        # those and not for the rest, and inventing span attributes nobody's dashboard
        # reads is how a span gets wide without getting more useful.
        if usage.input_tokens:
            span.set_attribute(semconv.INPUT_TOKENS, usage.input_tokens)
        if usage.output_tokens:
            span.set_attribute(semconv.OUTPUT_TOKENS, usage.output_tokens)

    def _record_cost(self, span: Span, outcome: Outcome, attributes: Mapping[str, Any]) -> None:
        """Price the call, if we know what it costs.

        Priced against the model that *served* the call where the provider named one,
        falling back to what was asked for. Those differ more often than you would like:
        aliases resolve, and a provider quietly serving a dearer model is precisely the
        thing this metric should be able to show you.
        """
        if outcome.usage is None:
            return

        model = outcome.model or str(attributes.get(semconv.TARGET, ""))
        dollars = cost_of(
            outcome.usage, provider=self._provider, model=model, pricing=self._pricing
        )

        if dollars is None:
            self._warn_unpriced(model)
            return

        if dollars:
            span.set_attribute(semconv.COST, dollars)
            get_meter().counter(semconv.COST, dollars, unit="USD", **attributes)

    def _warn_unpriced(self, model: str) -> None:
        """Say once that this model is missing from the price list, then be quiet.

        Silence here would be the worst outcome available: the cost dashboard simply has
        no bar for this model, which looks exactly like a model nobody is using.
        """
        if not model or model in self._unpriced:
            return

        self._unpriced.add(model)
        self._log.warning(
            "no price configured; tokens recorded, cost not",
            model=model,
            hint=f"add pricing.{self._provider}.{model} to price this component's calls",
        )

    def _record_failure(
        self,
        span: Span,
        exc: BaseException,
        operation: str,
        target: str,
        extra: Mapping[str, Any] | None,
        started: float,
    ) -> None:
        """Record a call that ran out of attempts, or was never going to get any.

        The exception class goes on the span and the metric; the message does not. A
        message is unbounded — it contains ids, model names, whole prompts sometimes —
        and an unbounded metric dimension is how you take down a metrics backend.
        """
        span.record_exception(exc)

        get_meter().histogram(
            semconv.DURATION,
            time.perf_counter() - started,
            unit="s",
            **self._attributes(operation, target, extra),
            **{semconv.ERROR_TYPE: type(exc).__name__},
        )
