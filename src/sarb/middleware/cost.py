"""Tokens times a price list.

That is the whole module, and it is deliberately this small. The hard part of cost is
not the arithmetic — it is knowing that the arithmetic is being fed the right numbers,
which is a problem solved in :class:`~sarb.contracts.Usage` and in the adapters that
normalise into it. By the time a :class:`~sarb.contracts.Usage` reaches this function,
"input tokens" means the same thing whichever vendor produced it, and multiplying is
allowed to be boring.

Two rules, both of them about not lying:

**An unpriced model has no cost, not a zero cost.** Zero is a claim — it says this call
was free — and a dashboard summing zeros reports a system that costs nothing to run.
``None`` says we do not know, which is the truth, and the caller records nothing. The
warning that goes with it is what turns "our cost dashboard is missing Claude" into a
one-line config fix instead of a week of confusion.

**An unstated cache price is the full price.** If the config prices a model's input but
says nothing about its cache, cache reads are billed at the input rate. That
*overstates* the bill for anyone using caching, which is the direction to be wrong in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from sarb.config import ModelPrice
    from sarb.contracts import Usage

__all__ = ["cost_of"]

#: Vendors quote per a million tokens, and the config holds what they quote.
PER = 1_000_000


def cost_of(
    usage: Usage,
    *,
    provider: str,
    model: str,
    pricing: dict[str, dict[str, ModelPrice]],
) -> float | None:
    """What one call cost, in US dollars.

    Args:
        usage: Normalised token counts. ``input_tokens`` must already include the cached
            and cache-written parts, which is what :class:`~sarb.contracts.Usage`
            validates.
        provider: The registry key the call went through — ``anthropic``, ``openai``.
        model: The model that served it. The *response* model where the provider said,
            because that is what is invoiced; an alias silently resolving to a dearer
            model is exactly the thing a cost metric should be able to show you.
        pricing: ``settings.pricing``, as ``pricing[provider][model]``.

    Returns:
        The cost in USD, or ``None`` if this model has no price configured. ``None`` and
        ``0.0`` are different answers and the caller must treat them differently: the
        first means *unknown*, the second means *free*, and a local model really is free.
    """
    price = pricing.get(provider, {}).get(model)
    if price is None:
        return None

    # Fall back to the full input price rather than to nothing. A config that priced the
    # model but forgot its cache should over-report, not under-report.
    cached = price.cached_input if price.cached_input is not None else price.input
    written = price.cache_write_input if price.cache_write_input is not None else price.input

    dollars = (
        usage.uncached_input_tokens * price.input
        + usage.cached_input_tokens * cached
        + usage.cache_write_input_tokens * written
        # Reasoning tokens are already inside output_tokens and are billed at the output
        # rate, so there is nothing to add here. Adding them would double-charge exactly
        # the calls that are already the most expensive.
        + usage.output_tokens * price.output
    )

    return dollars / PER
