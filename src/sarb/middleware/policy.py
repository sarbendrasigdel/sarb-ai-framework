"""When to try again, and how long to wait first.

Pure functions over a settings object. No clock, no sleeping, no I/O — which means
the backoff curve can be tested exhaustively, without a test suite that takes
thirty seconds to assert that it waited thirty seconds.
"""

from __future__ import annotations

import random

from sarb.config import ResilienceSettings
from sarb.errors import ProviderError, RateLimitError, SarbError

__all__ = ["backoff_for", "should_retry"]


def should_retry(exc: BaseException, *, attempt: int, policy: ResilienceSettings) -> bool:
    """Whether ``exc`` is worth another attempt.

    Args:
        exc: The failure.
        attempt: Which attempt just failed. 1-based.
        policy: The component's policy.

    Returns:
        ``True`` only when the error *says* it is retryable and attempts remain.

    The decision is the error's, not ours. Every adapter in this framework already
    classifies its failures — a ``429`` carries ``retryable=True``, a ``401`` carries
    ``retryable=False`` — so this function reads a verdict rather than reaching one.
    That is what lets it work on a provider nobody has written yet, including one from
    a plugin: it never inspects a status code, a vendor exception, or a message.

    Anything that is not a :class:`~sarb.errors.SarbError` is not retried. A
    ``TypeError`` from inside an adapter is a bug, and retrying a bug three times just
    means it takes three times as long to see it.
    """
    if attempt >= policy.max_attempts:
        return False

    if not isinstance(exc, SarbError):
        return False

    return bool(getattr(exc, "retryable", False))


def backoff_for(
    exc: BaseException,
    *,
    attempt: int,
    policy: ResilienceSettings,
    rng: random.Random | None = None,
) -> float:
    """How long to wait before the next attempt, in seconds.

    Args:
        exc: The failure that prompted the retry.
        attempt: Which attempt just failed. 1-based.
        policy: The component's policy.
        rng: Injected so the jitter is testable. Defaults to the shared generator.

    Returns:
        Seconds to sleep.

    A provider's own ``Retry-After`` wins outright, and is not capped by
    ``max_backoff``:
        It is an instruction from the only party that knows when the quota resets.
        Overriding it with a shorter wait means retrying into the same rate limit and
        being told again — having spent an attempt to learn nothing.

    Otherwise: exponential backoff with **full jitter** — a uniform draw from
    ``[0, min(cap, base * 2**n))`` rather than that value exactly:
        This is not decoration. When an indexing job fires forty concurrent embedding
        batches and all forty are rate-limited in the same instant, an unjittered
        backoff retries all forty at precisely ``t+1s`` — recreating the burst that
        caused the problem, in lockstep, until the attempts run out. Spreading them
        across the window is the difference between recovering and oscillating.
    """
    if isinstance(exc, RateLimitError) and exc.retry_after is not None:
        return exc.retry_after

    if isinstance(exc, ProviderError):
        # Some providers report Retry-After on errors that are not rate limits —
        # a 503 during a deploy, for instance. Honour it wherever it appears.
        asked_for = getattr(exc, "retry_after", None)
        if isinstance(asked_for, int | float):
            return float(asked_for)

    ceiling = min(policy.max_backoff, policy.initial_backoff * 2 ** (attempt - 1))
    return (rng or random).uniform(0.0, ceiling)
