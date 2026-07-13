# Embedding providers

| `provider:` | Extra | What it is |
| --- | --- | --- |
| `openai` | `sarb[openai]` | OpenAI embeddings — and, via `base_url`, any API-compatible server (TEI, vLLM, LiteLLM, Together, Jina) |
| `fastembed` | `sarb[fastembed]` | Local ONNX on the CPU. No key, no GPU, no torch |
| `sentence-transformers` | `sarb[sentence-transformers]` | Local PyTorch. GPU, and any fine-tune on the Hub |
| `fake` | — | Deterministic vectors from a hash. No model at all |

```yaml
embedders:
  default:
    provider: fastembed
    options:
      model: BAAI/bge-small-en-v1.5
```

```python
embedder = await container.get(EMBEDDERS)

indexed = await embedder.embed(chunks)                        # indexing
asked = await embedder.embed([q], kind=EmbeddingKind.QUERY)   # searching

indexed.vectors   # list[Vector] — one per input, in order
indexed.usage     # tokens, and therefore cost. Zero for a local model
```

`embed()` returns an [`EmbeddingResponse`](usage-and-cost.md), not a bare list. The extra
attribute access buys a token count: without it, embedding was the one priced operation
in the framework that no dashboard could see, and a full re-index of a corpus — the
largest embedding bill anyone ever gets — was invisible.

The adapter shape is the one Module 5 established — options schema, adapter with its
client injected, `build()` doing the lazy import, a line in the registry — so this
page covers only what is genuinely different about embedding. Read
[LLM providers](llm-providers.md) first if you have not.

## Embedding fails silently, and that shapes everything here

The other ports fail loudly. A bad chat call raises. A bad vector-store query raises.

Embedding does not. Its failure mode is a vector that is the wrong width, or in the
wrong *position*, written happily into an index that will not complain — and the only
symptom is that retrieval is somewhat worse than it should be, forever, with no
traceback anyone will ever find. Nobody can even say when it started.

So this module is more suspicious than its size suggests, in three specific places.

### The output is checked against the port's promise, on every call

The port says: *one vector per input, in the same order, always.* `check_output()`
verifies exactly that, and it is not decoration.

If a provider returns fewer vectors than it was given texts, every vector after the
gap is stored against the *wrong chunk*. The index still works. Every chunk in it is
now findable only by the text of some other chunk. Nothing raises.

If a vector comes back the wrong width, the collection it is bound for was created
with a different one — which is what a model change that was not carried through to
the vector store looks like, and also what a hosted API quietly ignoring the
`dimensions` it was sent looks like. Both are one deploy away.

```
EmbeddingError: provider returned a 1536-dimensional vector where 384 was configured
```

### Order is taken from the response, not from the wire

The OpenAI adapter sorts its response by the API's own `index` field rather than
trusting the array to arrive in order. That field exists precisely because order is a
promise the transport makes and not one the protocol does — and a permuted batch is
the exact silent failure above.

It costs a sort of a few hundred items. It buys never having to wonder.

### `EmbeddingKind` is honoured, differently, by each adapter

Many strong retrieval models — BGE, E5, most instruction-tuned embedders — are
**asymmetric**: they expect a different prefix on the thing being searched for than on
the thing being searched. Embed a query as a document and nothing errors. The vector
just lands slightly off, recall quietly drops, and the system looks like it works.

That is why `kind` is on the port at all, and why the caller must state it. Each
adapter then does what its model needs:

| Adapter | How |
| --- | --- |
| `fastembed` | `query_embed()` versus `embed()` — the library owns the distinction |
| `sentence-transformers` | a `prompt` prefix, which is **configuration** — see below |
| `openai` | ignores it. Its embedding models are symmetric, and the port permits this explicitly |
| `fake` | records it, so a test can assert the pipeline got it right |

The prefixes are config because they have to be. E5 wants `"query: "` and
`"passage: "`. BGE wants an instruction on the query and nothing on the document. Many
models want neither, and prefixing those makes retrieval *worse*. We cannot know which
model you loaded, and a table of model names to prompts is a table that is wrong the
week after it is written — so we ask:

```yaml
embedders:
  default:
    provider: sentence-transformers
    options:
      model: intfloat/e5-small-v2
      query_prompt: "query: "
      document_prompt: "passage: "
```

Both default to empty, which is right for a symmetric model and wrong for an
asymmetric one, silently. That is uncomfortable, and it is still the honest option: the
alternative is guessing, and a guess here costs recall that nobody will ever trace back
to this line.

## `dimensions`: required when hosted, discovered when local

The vector width is not a detail. It is what the vector-store collection gets created
with, so it must be knowable **before** anything is embedded — you cannot find it out
by embedding something, because by then the collection exists at whatever width
somebody guessed.

**Hosted (`openai`): you state it, and it is required.**

```yaml
options:
  model: text-embedding-3-small
  dimensions: 1536
```

This is not the framework being lazy. With `text-embedding-3-*` the width genuinely
*is* a request parameter — the vectors are Matryoshka embeddings and can be truncated
to whatever you ask for — so there is no single right answer to default to. Stating it
also makes it binding: the adapter sends it, the API truncates to it, and
`check_output` confirms what came back.

`send_dimensions: false` for a server that rejects the parameter (`ada-002`, and most
self-hosted ones). The width check still runs, so a wrong value fails on the first call
rather than corrupting an index.

**Local: the library is asked.**

FastEmbed publishes the widths of the models it supports, and
`SentenceTransformer.get_sentence_embedding_dimension()` answers once the model is
loaded. So the adapter reads the width from the library rather than from a table of its
own — a vendor is allowed to know about its own models; *we* are not, because our copy
of that knowledge goes stale in silence. Same rule that kept context windows out of
Module 5.

## Local models block, and that is a production problem

An HTTP adapter awaits. A local model does not: it is a CPU- or GPU-bound forward pass
in this process, and calling it directly from the event loop stalls **everything** —
every other request, every open stream, every health check — for as long as it takes.
For a batch of documents that is not milliseconds.

So both local adapters run their work in a worker thread. That fixes the stall and
introduces the opposite problem, which is why the limiter is not optional:

```yaml
options:
  max_concurrency: 1     # the default
```

One model object, one set of weights. Running fifty forward passes at once does not
make any of them faster; it multiplies peak memory by fifty and takes the process down.
Queuing is the correct behaviour here, it is *not* what `to_thread` does by default, so
it is stated rather than assumed — and asserted, in `test_shared.py`.

Relatedly: **local models load in `build()`, at startup.** Loading is slow and touches
the network. Both of those are better in front of the person deploying than in front of
the first user, and a mistyped model name should fail the boot, not the first search.

## The fake embedder

`provider: fake` is a registered provider, not a test fixture. Put it in the
`embedders` block and the whole application — indexing, retrieval, the API — runs on a
laptop with nothing installed and no key.

Its vectors are seeded from a SHA-256 of the text, so the same text yields the same
vector in this process and every future one. (Not `hash()`, whose salt changes per
process: a "deterministic" fake that answers differently on the next run is worse than
an obviously random one, because it makes a flaky test look like a real failure.)

Two features earn their keep in engine tests:

```python
embedder = FakeEmbeddingProvider(FakeEmbeddingOptions(
    dimensions=2,
    vectors={"Paris is the capital": (0.0, 1.0)},   # place things deliberately
))

await pipeline.index(documents)
await retriever.search("what is the capital?")

assert embedder.kinds == [DOCUMENT, DOCUMENT, QUERY]   # ← the assertion that matters
```

`vectors` scripts the geometry, so a retrieval test can say *these chunks are near this
query* and then assert the ranking that must follow — hashed vectors are deterministic
but arbitrary, and you cannot ask for "a document near this query" without knowing where
the hash happened to put it.

`kinds` records how every text was embedded. An indexing pipeline that embeds a query as
a document produces a system that *works*, just worse, and no test that only inspects
the vectors will ever catch it. One that inspects `kinds` catches it immediately.

## The status table moved up a level

`error_for_status`, `retry_after_seconds` and `unsupported` used to live in
`providers/llm/_shared.py`. They now live in `providers/_shared.py`, with the fallback
error class passed in:

```python
error_for_status(status, message, provider=..., model=..., otherwise=EmbeddingError)
```

A `429` means the same thing whether it interrupted a chat completion or a batch of
embeddings, and a backoff policy that had to ask *which kind* of 429 that was would be
the wrong abstraction. Copying the table into each family would have recreated exactly
the "one bug in every provider at once" problem the shared module exists to prevent —
in the error path, which is the one path nobody exercises.

The *kind* of call is the caller's knowledge. The *meaning* of the status is not.

## API reference

::: sarb.providers.embedding
