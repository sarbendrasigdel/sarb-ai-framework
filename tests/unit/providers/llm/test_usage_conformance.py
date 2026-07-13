"""One question, every vendor, the same answer.

This suite exists because of a bug that shipped, survived three modules of review, and
would have been found by nobody until an invoice arrived.

Every LLM adapter has a ``_usage()`` that copies a vendor's token counts into our
:class:`~sarb.contracts.Usage`. Each one looked obviously correct. Each one was tested,
against a response built from that vendor's own SDK models. And they still disagreed
with each other, because the vendors disagree with each other about what the *words*
mean:

* OpenAI's ``prompt_tokens`` **includes** the tokens it also reports as cached.
* Anthropic's ``input_tokens`` **excludes** them: ``total = input + read + creation``.
* Gemini bills reasoning as *"the sum of output tokens and thinking tokens"* — its
  thinking tokens sit **outside** ``candidates_token_count``, where OpenAI's and
  Anthropic's sit inside their output counts.

So ``Usage.input_tokens`` meant three different things depending on who served the call,
and no per-adapter test could ever have said so — each was checking its adapter against
the same misunderstanding that produced it. The only test that catches this is one that
asks *all* of them the same question and demands one answer.

This is the same argument, and the same shape, as the vector-store conformance suite.
When a fourth provider is added, it goes in ``CASES`` and it either agrees or it is
wrong.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from anthropic.types import Usage as AnthropicUsage

from sarb.contracts import Usage
from sarb.providers.llm.anthropic import AnthropicProvider
from sarb.providers.llm.gemini import GeminiProvider
from sarb.providers.llm.openai import OpenAIProvider

# --------------------------------------------------------------------------- #
# The scenario, in each vendor's own words
# --------------------------------------------------------------------------- #
#
# Every case below describes the *same call*:
#
#   1,000 input tokens, of which 600 were read from a cache
#   500 output tokens, of which 200 were spent reasoning
#
# What differs is only how each vendor writes that down. Which is the whole point.


def anthropic_usage():
    """Anthropic: input_tokens is only what is left *after* the cache breakpoints."""
    vendor = AnthropicUsage(
        input_tokens=300,  # fresh only — the other 700 are below
        output_tokens=500,  # thinking already inside
        cache_read_input_tokens=600,
        cache_creation_input_tokens=100,  # the one vendor here that reports these
    )
    vendor.output_tokens_details = type("D", (), {"reasoning_tokens": 200})()

    return AnthropicProvider._usage(vendor)


def openai_usage():
    """OpenAI: prompt_tokens is the total, cache included. No cache-write reporting."""
    vendor = type(
        "U",
        (),
        {
            "prompt_tokens": 1_000,  # total, cache included
            "completion_tokens": 500,  # reasoning already inside
            "prompt_tokens_details": type("P", (), {"cached_tokens": 600})(),
            "completion_tokens_details": type("C", (), {"reasoning_tokens": 200})(),
        },
    )()

    return OpenAIProvider._usage(vendor)


def gemini_usage():
    """Gemini: thinking tokens are reported *beside* the answer's, not within them."""
    vendor = type(
        "U",
        (),
        {
            "prompt_token_count": 1_000,  # total, cache included
            "candidates_token_count": 300,  # the answer ONLY — 500 - 200 thinking
            "cached_content_token_count": 600,
            "thoughts_token_count": 200,
        },
    )()

    return GeminiProvider._usage(vendor)


#: Only Anthropic reports cache *writes*, because only Anthropic makes you ask for the
#: cache: OpenAI's is automatic and Gemini's storage is billed by the hour rather than by
#: the token, so neither has a write to attribute and the field stays zero for both.
#:
#: That is a genuine difference in what the vendor knows, not a difference in what the
#: fields mean, so it is declared here as data rather than papered over. Each case says
#: how many of its thousand input tokens it can account for as cache writes.
CASES = [
    pytest.param(anthropic_usage, 100, id="anthropic"),
    pytest.param(openai_usage, 0, id="openai"),
    pytest.param(gemini_usage, 0, id="gemini"),
]


@pytest.mark.parametrize(("translate", "cache_writes"), CASES)
class TestEveryVendorMeansTheSameThing:
    def test_the_totals_are_totals(self, translate, cache_writes):
        # The load-bearing assertion. All three vendors just described a call with a
        # thousand input tokens; only one of them said "1000".
        usage = translate()

        assert usage.input_tokens == 1_000
        assert usage.output_tokens == 500

    def test_the_cached_part_is_inside_the_input_total(self, translate, cache_writes):
        usage = translate()

        assert usage.cached_input_tokens == 600
        assert usage.cache_write_input_tokens == cache_writes
        assert usage.cached_input_tokens <= usage.input_tokens

    def test_the_reasoning_part_is_inside_the_output_total(self, translate, cache_writes):
        # Gemini's is the one that has to be *added* to get here. Miss it and every
        # reasoning call under-reports output by the part that costs the most.
        usage = translate()

        assert usage.reasoning_tokens == 200
        assert usage.reasoning_tokens <= usage.output_tokens

    def test_the_full_price_input_is_what_is_left(self, translate, cache_writes):
        # What cost actually multiplies. 1000 - 600 cached - whatever was written.
        usage = translate()

        assert usage.uncached_input_tokens == 1_000 - 600 - cache_writes


class TestTheContractIsEnforcedNotMerelyDocumented:
    def test_a_cache_larger_than_the_input_it_is_part_of_is_rejected(self):
        # Exactly the shape of the Anthropic bug: pass the vendor's numbers through
        # unchanged and, on a cache hit, `cached` exceeds `input`. Without this validator
        # nothing downstream notices — cost simply bills a negative number of fresh
        # tokens, and a negative dollar figure sums quietly into a total that is merely
        # a bit low.
        with pytest.raises(ValueError, match="must include them both"):
            Usage(input_tokens=300, cached_input_tokens=600)

    def test_reasoning_larger_than_the_output_it_is_part_of_is_rejected(self):
        # And the shape of the Gemini one.
        with pytest.raises(ValueError, match="must include them"):
            Usage(output_tokens=300, reasoning_tokens=500)
