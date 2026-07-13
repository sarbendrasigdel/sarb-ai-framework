# The ports

Eleven ports, one per replaceable abstraction, each resolved from one config
family.

| Port | Family | Abstracts |
| --- | --- | --- |
| `LLMProvider` | `llms` | language models |
| `EmbeddingProvider` | `embedders` | text → vectors |
| `VectorStore` | `vector_stores` | vector storage and search |
| `MemoryProvider` | `memories` | conversation history |
| `PromptProvider` | `prompts` | versioned prompts |
| `DocumentLoader` | `loaders` | source → documents |
| `ChunkingStrategy` | `chunkers` | document → chunks |
| `RetrievalStrategy` | `retrievers` | question → passages |
| `ToolProvider` | `tools` | callable tools |
| `WorkflowExecutor` | `workflows` | step orchestration |
| `StorageProvider` | `storages` | blobs |

Every one is a `typing.Protocol`. A class satisfies a port by **shape** — no
inheritance, no `sarb` import. An [architecture test](overview.md) asserts exactly
that, by implementing all eleven with plain classes that import nothing.

```python
llm = await container.get(LLMS)              # -> LLMProvider
store = await container.get(VECTOR_STORES)   # -> VectorStore
judge = await container.get(LLMS, "judge")   # by name
```

---

## The decisions worth arguing about

These are the choices that are expensive to reverse. Each one had a plausible
alternative that was rejected for a stated reason.

### 1. The system prompt is a field, not a message

```python
LLMRequest(system="You are a careful analyst.", messages=(Message.user("..."),))
```

`Role` has `USER`, `ASSISTANT`, `TOOL` — and deliberately **no `SYSTEM`**.

Anthropic takes the system prompt as a top-level parameter; OpenAI takes it as a
message. Either shape can render the other, so the question is which one prevents
more mistakes. A field does: it makes *"a system message halfway through a
conversation"* unrepresentable, because that is not a meaningful thing to ask for,
and it means the framework never has two ways to say the same thing.

**The cost:** conversation histories that already store a system message (imported
from an OpenAI-shaped log, say) need a migration step at the boundary.

### 2. Tool calls are fields; tool *results* are one message, not many

```python
message.tool_calls      # on assistant messages
message.tool_results    # on tool messages — plural
```

Anthropic models tool use as blocks *inside* content; OpenAI as a field on the
message. Content-parts would have been the more "general" choice — and it is what
Anthropic does — but the code that actually consumes this is the agent loop, where
`if message.tool_calls:` is plainly better than scanning a content sequence for
parts of a particular kind.

Models emit **parallel** tool calls. OpenAI wants one message per result; Anthropic
wants one message containing all of them. Modelling one message with a *tuple* of
results keeps the framework's own history honest — one model turn, one response
turn — and leaves the fan-out to the OpenAI adapter, which is what an adapter is
for.

### 3. Malformed tool arguments are carried, not raised

```python
class ToolCall:
    arguments: dict[str, Any]
    malformed_arguments: str | None   # the raw string, when it wasn't valid JSON
```

Models do occasionally emit broken JSON. Both obvious designs fail: **raising**
turns a recoverable event into a dead request, and **coercing to `{}`** invents an
intent the model never had. Carrying the raw text lets the tool engine hand the
model back an error it can retry from — which is what a competent colleague would
do, and what the model is entirely capable of doing.

### 4. `EmbeddingKind` is required on every embed call

```python
await embedder.embed(texts, kind=EmbeddingKind.QUERY)
```

Most strong retrieval models — BGE, E5, most instruction-tuned embedders — are
**asymmetric**: they expect a different prefix for the thing being searched *for*
than for the thing being searched. Embed a query as a document and nothing errors.
Recall just quietly drops, and the system looks like it works.

That bug is invisible, and it is exactly the bug a framework whose whole premise is
"the embedding model is swappable" will cause. Making the caller say which one they
mean is the only defence. Symmetric models ignore it.

### 5. Filters are a small algebra, not a passthrough dict

```python
where = eq("tenant", "acme") & gte("year", 2020) & ~exists("archived")
```

Similarity search is easy to abstract; **metadata filtering** is where vendor
neutrality is actually won or lost. Most "pluggable vector store" abstractions
quietly give up here and expose a `dict` passed straight to the backend — and the
moment an application writes one, it is married to Qdrant forever.

So: `eq · ne · gt · gte · lt · lte · in · not_in · exists`, composed with
`& | ~`. Small enough that Qdrant's DSL and SQL can both express all of it. It
serialises to JSON, so it can be stored, logged, and audited.

**An adapter that cannot express a filter must raise, never ignore.** A dropped
filter returns *more* results, not fewer — it fails **open**, silently, across
whatever boundary the filter was enforcing. Usually tenant isolation.

### 6. `SearchHit.score`: higher is always better

Euclidean distance is a *distance* — lower is better. An adapter using it must
invert before reporting. Normalising the direction at the port means retrieval code
sorts one way, always, without asking which metric the store happens to use.

### 7. Chunks carry character offsets

```python
Chunk(document_id="d1", text="...", index=4, start_char=4120, end_char=4890)
```

This is what makes a citation **verifiable**. A RAG system that can only say "this
came from report.pdf" is asking the user to trust it; one that can say "characters
4120–4890 of report.pdf" lets them check.

Offsets are easy to drop and nearly impossible to add back later — by then a dozen
chunkers have shipped without them.

### 8. `StreamEnd` carries the fully assembled response

```python
async for event in llm.stream(request):
    match event:
        case TextDelta(text=t): print(t, end="")
        case StreamEnd(response=r): history.append(r.message)
```

Without it, **every** consumer that needs the whole message — to append to history,
to dispatch tool calls, to count tokens — reassembles it from deltas, and every one
does it slightly differently. The adapter assembles it once, from its provider's own
terminal event, which it is far better placed to do correctly.

### 9. `MemoryProvider` is only conversation history

"Memory" is used for three different things: history, a rolling summary, and
semantic recall of facts. They have genuinely different shapes — an append-only log,
a derived value, a search index — and one port spanning all three degenerates into
`get(key)` / `set(key, value)`, which abstracts nothing.

So `MemoryProvider` is the **append-only log, only**. Summarisation is a strategy
layered on top (a provider that summarises on write still satisfies this port).
Semantic recall is `RetrievalStrategy`, which already exists and already does that
job.

### 10. `TokenCounter` is a separate port

Only some providers can count tokens locally. Putting `count_tokens` on
`LLMProvider` would force every adapter to implement a method most of them can only
fake. Interface segregation:

```python
if isinstance(llm, TokenCounter):
    ...
```

### 11. Workflow is the least settled port — and says so

The other ten are anchored by an external reality — an LLM API, a vector database, a
filesystem — which sharply limits how wrong their shape can be. Workflow
orchestration has no such anchor: the design space runs from "a list of steps" to a
full actor system.

So `Workflow` is deliberately the **smallest useful thing**: a validated DAG of
steps, each naming a component to invoke. Two things it explicitly does *not*
attempt, because half-doing them is worse than not doing them:

- **Durable execution** — resuming after a process dies. A genuinely hard
  distributed-systems problem; the honest options are to use something built for it,
  or to accept in-memory runs.
- **Cycles.** A DAG is acyclic. Iteration belongs in the agent engine, which has a
  turn limit and a model deciding when to stop.

Cycles, dangling dependencies and duplicate ids are rejected when a workflow is
**loaded**, not when it runs — a cycle discovered half way through a pipeline has
already spent real money on model calls and may already have written to a real index.

When a real requirement outgrows this shape, it should be **replaced**, not extended
into a Turing machine one field at a time. It is small precisely so that stays cheap.

---

## Known limitations

Stated rather than discovered later:

- **`ToolResult.content` is text.** Tools returning images (screenshots, charts) will
  need a parts sequence, exactly as `Message` has. The change is additive; deferred
  until there is a caller.
- **No hybrid/sparse search in `VectorQuery`.** `RetrievalStrategy` is the seam where
  hybrid search belongs, and it can compose keyword search itself. If a *store*-level
  sparse vector proves necessary, it is an additive field.
- **`Usage` has no cost.** Cost is tokens × a price list that changes without warning
  and differs per contract. Putting a number in the domain model guarantees it is
  wrong the day a vendor updates its pricing page.

## API reference

::: sarb.contracts
