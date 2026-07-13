# Token usage and cost

```yaml
pricing:
  anthropic:
    claude-sonnet-5:
      input: 3.00 # USD per MILLION tokens — what the pricing page says
      output: 15.00
      cached_input: 0.30
      cache_write_input: 3.75
```

That is the whole of the configuration. Everything else on this page is about the part
that is genuinely hard, which is not the multiplication.

## The multiplication was never the problem

Cost is tokens times a price. The arithmetic is a line of code, and
[`middleware/cost.py`](#api-reference) is about forty lines including the reasons.

The problem is that **the tokens were wrong**, and wrong in a way that nothing could
see. Every adapter in this framework carefully read its vendor's token counts and copied
them into `Usage`. Every one of them looked correct. Every one had a test, written
against that vendor's own SDK models. And they disagreed with each other, because the
vendors disagree about what the words mean:

| | What the vendor's "input tokens" counts | What its "reasoning tokens" are inside |
| --- | --- | --- |
| **OpenAI** | Everything, cache **included** | `completion_tokens` |
| **Anthropic** | Only what is *left over* — cache **excluded** | `output_tokens` |
| **Gemini** | Everything, cache **included** | **Nothing.** Billed *beside* the output |

Anthropic's documentation is explicit: `total_input_tokens = cache_read_input_tokens +
cache_creation_input_tokens + input_tokens`. Google's pricing wording is equally
explicit: *"response pricing is the sum of output tokens and thinking tokens"*.

So `Usage.input_tokens` meant three different things depending on who served the call.
Cost computed from it would have been right for OpenAI, low for Anthropic by exactly the
cached portion, and low for Gemini by exactly the reasoning — which is the most expensive
part of the most expensive calls.

Nothing would have caught this. Each adapter's tests agreed with the adapter, because
both were written from the same misreading.

## So the contract is pinned, and validated

`Usage` now states what its fields mean, and the totals are **inclusive**:

```python
input_tokens              # ALL input: cache reads and writes included
├── cached_input_tokens       # of those, served from cache (≈ a tenth of the price)
└── cache_write_input_tokens  # of those, written to cache (dearer than fresh)

output_tokens             # ALL generated: reasoning included
└── reasoning_tokens          # of those, spent thinking (billed as output)
```

The adapter does the arithmetic to get there. Anthropic's adds its three input numbers
together; Gemini's adds thinking into output; OpenAI's, whose conventions happen to be
ours, does nothing at all — and says so in a comment, because in a file where the other
two do arithmetic at that exact point, the absence of any looks like an oversight.

And it is **checked**, not merely documented. A `Usage` whose cached tokens exceed its
input tokens will not construct:

```
ValueError: cached (600) + cache writes (0) exceed input_tokens (300).
            input_tokens must include them both.
```

That is precisely the shape of the Anthropic bug. Without the validator it produces no
error at all — cost simply bills a *negative* number of fresh tokens, and a negative
dollar figure sums quietly into a total that is merely a bit low.

`tests/unit/providers/llm/test_usage_conformance.py` describes one call — a thousand
input tokens, six hundred of them cached; five hundred output, two hundred of them
thinking — in each vendor's own dialect, and demands that all three adapters produce the
same `Usage`. It is the same argument as the vector-store conformance suite, and it
exists for the same reason: **a test written by the person who misunderstood the spec
will agree with the code they wrote from that misunderstanding.**

## Streaming used to record nothing at all

The second hole, and the larger one in dollars.

`Guard.run()` took an `outcome` callback and recorded tokens from it. `Guard.stream()`
had no such parameter, so a streamed call recorded a span, recorded a duration, and
recorded **no tokens and no cost whatsoever**.

It was invisible because it was not an error. The dashboards existed. The graphs
rendered. The numbers on them were simply absent — and since anything with a chat
interface streams, the absent numbers were most of the bill.

The fix is a hook that reads
[`StreamEnd`](../architecture/ports.md), which is why that event carries a fully
assembled response in the first place:

```python
def _stream_outcome(event: StreamEvent) -> Outcome | None:
    if isinstance(event, StreamEnd):
        return _outcome(event.response)
    return None   # every other event is a fragment and knows nothing about totals
```

A streamed call and a unary call now cost the same, which they always did — the invoice
never cared how the answer was delivered.

## What gets recorded

One counter and one histogram, on every call, through the guard that already wraps every
component:

| Metric | Instrument | Split by |
| --- | --- | --- |
| `gen_ai.client.token.usage` | Histogram | `gen_ai.token.type`: `input`, `output`, `cached_input`, `cache_write_input`, `reasoning` |
| `sarb.cost.usd` | Counter | `sarb.component`, `gen_ai.provider.name`, `gen_ai.request.model` |

The token types **overlap on purpose**: `cached_input` is *inside* `input`, not beside
it. Summing across the dimension would double-count, and that is the correct trade — the
dimension exists so a dashboard can *filter*, and a dashboard that wants the total
already has `input`. The alternative is a metric you must do arithmetic on to avoid being
lied to.

Cost is a counter rather than a histogram because the question is *"what did we spend"*,
and that is a sum.

`sarb.component` is the attribute you will actually group by. An application runs three
models and two are from the same vendor; `gen_ai.provider.name` cannot tell `llms.judge`
from `llms.default`, and the config already gave them names.

## Two rules about not lying

**An unpriced model has no cost, not a zero cost.** A model missing from the `pricing`
block records its tokens and no dollars, and warns once:

```
WARNING no price configured; tokens recorded, cost not
        model=claude-opus-4-8 hint=add pricing.anthropic.claude-opus-4-8 …
```

Recording `0.0` would be a *claim* — that this call was free — and a dashboard summing
zeroes reports a system that costs nothing to run. That is a far more comfortable lie
than an empty graph, and therefore a far more dangerous one. Once, not per call, because
a warning on every call is a warning everybody filters out of their logs.

A local model genuinely is free, and says so with a price of zero. `None` and `0.0` are
different answers to different questions.

**An unstated cache price is the full price.** If you price a model's input but say
nothing about its cache, cache reads are billed at the input rate. That *overstates* what
caching costs you rather than understating it — the framework helpfully assuming "oh,
that'll be about a tenth" is how a bill arrives that nobody forecast. Surprises should be
pleasant ones.

## Cost is priced against the model that *served* the call

Not the one you asked for. Providers alias, and an alias that quietly resolves to a
dearer model is exactly the thing a cost metric should be able to show you — so
`gen_ai.response.model` wins over `gen_ai.request.model` wherever the provider names one.

## Why the price list is configuration

Because it is the single most perishable fact in the system. Vendors cut prices, add
tiers, and sign contracts that make your rate different from the published one — and none
of that should require a release of a framework.

It is nested under the provider because a model name is not unique. An `llama3` served by
Ollama on hardware you already own is free. The same name behind somebody's hosted API is
not. `pricing[provider][model]` is unambiguous; a flat table of model names is a bug
waiting for the first company to self-host something.

Prices are **per million tokens**, matching every vendor's pricing page, so the config is
filled in by copying the numbers you are looking at. Per-token would mean transcribing
`0.000003`, and one misplaced zero is a cost dashboard that is wrong by a factor of ten
with nothing whatsoever to show for it.

## What is still missing

**Attribution.** Every metric carries the component, the provider and the model. None of
them carries a tenant, a user, or a feature — so *"which customer is costing us money"*
remains unanswerable no matter how good the token counts are. That needs a request-scoped
context, which needs the request scope, which arrives with the API module.

**Enforcement.** Nothing here refuses a call that would exceed a budget. Measuring and
enforcing are different jobs, and conflating them gives you a metrics pipeline that can
fail your requests. Budgets need Redis and a scope; they will get their own module.

## API reference

::: sarb.middleware.cost
