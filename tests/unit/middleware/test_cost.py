"""Tokens times a price list.

The arithmetic is trivial and the ways of getting it wrong are not, so each of them gets
a test: double-charging reasoning, forgetting that cached tokens are *inside* the input
total, and reporting an unpriced model as free.
"""

from __future__ import annotations

from sarb.config import ModelPrice
from sarb.contracts import Usage
from sarb.middleware.cost import cost_of

#: $3 in, $15 out, per million — a plausible mid-range model, with the usual cache rates.
PRICING = {
    "anthropic": {
        "claude-sonnet-5": ModelPrice(
            input=3.0, output=15.0, cached_input=0.30, cache_write_input=3.75
        ),
        "no-cache-prices": ModelPrice(input=3.0, output=15.0),
    },
    "ollama": {"llama3": ModelPrice()},  # runs on your own hardware; genuinely free
}


def usd(usage: Usage, *, provider: str = "anthropic", model: str = "claude-sonnet-5"):
    return cost_of(usage, provider=provider, model=model, pricing=PRICING)


class TestTheArithmetic:
    def test_a_plain_call(self):
        # 1M in at $3, 1M out at $15.
        assert usd(Usage(input_tokens=1_000_000, output_tokens=1_000_000)) == 18.0

    def test_cached_tokens_are_discounted_not_added(self):
        # The subtlety the whole `Usage` contract exists to protect. `input_tokens` is
        # the *total*, so the cached part must be priced at the cache rate and the rest
        # at the full rate — not the total at full rate *plus* the cache again.
        usage = Usage(input_tokens=1_000_000, cached_input_tokens=900_000)

        # 100k fresh at $3/M + 900k cached at $0.30/M = $0.30 + $0.27
        assert usd(usage) == 0.57

        # Which is the point of caching: the same million tokens cost $3.00 without it.
        assert usd(Usage(input_tokens=1_000_000)) == 3.0

    def test_cache_writes_cost_more_than_fresh_input(self):
        # You pay a premium now to pay a tenth of it later. A cost model that missed this
        # would report the first call of a cached conversation as cheaper than it was.
        usage = Usage(input_tokens=1_000_000, cache_write_input_tokens=1_000_000)

        assert usd(usage) == 3.75

    def test_reasoning_tokens_are_not_charged_twice(self):
        # They are *inside* output_tokens and billed at the output rate. Adding them
        # again would overcharge exactly the calls that are already the most expensive,
        # which is the worst place to put a factor-of-two error.
        thinking = Usage(input_tokens=0, output_tokens=1_000_000, reasoning_tokens=800_000)
        plain = Usage(input_tokens=0, output_tokens=1_000_000)

        assert usd(thinking) == usd(plain) == 15.0

    def test_a_free_model_costs_zero(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)

        assert usd(usage, provider="ollama", model="llama3") == 0.0


class TestWhatIsNotKnown:
    def test_an_unpriced_model_has_no_cost_rather_than_a_zero_cost(self):
        # `None` and `0.0` are different claims and the caller treats them differently.
        # Zero says *this call was free*, and a dashboard summing zeroes reports a system
        # that costs nothing to run — which is a far more comfortable lie than an empty
        # graph, and therefore a far more dangerous one.
        assert usd(Usage(input_tokens=1_000), model="claude-opus-4-8") is None

    def test_an_unknown_provider_is_unpriced_too(self):
        assert usd(Usage(input_tokens=1_000), provider="who") is None

    def test_the_same_model_name_can_cost_different_things_at_different_providers(self):
        # Which is why the price list is nested under the provider. `llama3` on your own
        # hardware is free; the same name behind somebody's hosted API is not.
        assert usd(Usage(input_tokens=1_000_000), provider="ollama", model="llama3") == 0.0
        assert usd(Usage(input_tokens=1_000_000), provider="anthropic", model="llama3") is None

    def test_an_unstated_cache_price_falls_back_to_the_full_input_price(self):
        # Overstating the bill is the direction to be wrong in. If the config prices a
        # model but says nothing about its cache, the framework must not helpfully assume
        # a tenth — the surprise that costs you is the one where the real invoice is
        # *larger* than the forecast.
        usage = Usage(input_tokens=1_000_000, cached_input_tokens=1_000_000)

        assert usd(usage, model="no-cache-prices") == 3.0
