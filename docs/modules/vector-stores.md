# Vector stores

| `provider:` | Extra | What it is |
| --- | --- | --- |
| `qdrant` | `sarb[qdrant]` | A purpose-built vector database. What you reach for when the corpus is large |
| `pgvector` | `sarb[pgvector]` | Vectors in the Postgres you already run. One database, one backup, one transaction |
| `memory` | — | Exact search, in this process. No server, no Docker, no network |

```yaml
vector_stores:
  default:
    provider: qdrant
    options:
      url: ${QDRANT_URL:-http://localhost:6333}
      collection: documents
```

```python
store = await container.get(VECTOR_STORES)

await store.ensure_collection("documents", dimensions=384, distance=Distance.COSINE)
await store.upsert("documents", records)

hits = await store.search(
    "documents",
    VectorQuery(vector=q, top_k=5, where=eq("tenant", "acme") & gte("year", 2024)),
)
```

The adapter shape is the one Module 5 established and this page does not repeat it. What
is genuinely new here is the **filter algebra** — the first thing in the framework that
one adapter must *compile* rather than merely call — and almost everything below is a
consequence of that.

## Why there is a filter algebra at all

Every vector store can filter. No two of them agree on how to say so.

Qdrant wants a tree of `FieldCondition` objects. Postgres wants a SQL predicate over a
JSONB column. Weaviate wants `GraphQL`. Pinecone wants a Mongo-shaped dict. If the port
took "whatever your backend's filter language is", then the first retrieval engine
written against it would contain a Qdrant filter, and the port would have been a lie
from the first commit — swapping the store would mean rewriting the engine.

So the port takes *our* filter, and each adapter compiles it:

```python
eq("tenant", "acme") & (gte("year", 2024) | exists("pinned"))
```

`Compare`, `And`, `Or`, `Not`, and nine operators — `eq`, `ne`, `gt`, `gte`, `lt`,
`lte`, `in`, `not_in`, `exists`. Deliberately small. It is the intersection of what
every backend can express, not the union: an algebra that can say something only Qdrant
can hear is an algebra that has quietly picked a vendor.

### An adapter that cannot express a filter must **raise**

This is the most important rule in the module, and it is worth being blunt about why.

A filter is usually a security boundary. `eq("tenant", "acme")` is very often the only
thing standing between one customer's documents and another's. So consider what a
dropped filter does: the query still runs, it still returns plausible-looking documents,
it is still fast, and nothing anywhere raises. It just returns *more* than it should.

A dropped filter **fails open**. That is the one failure mode a retrieval system must
never have, and it is invisible in every test that only checks that results came back.

So no adapter is permitted to do its best. If it cannot compile a filter faithfully, it
raises and the query does not run:

```python
lt("title", "m")   # Qdrant's Range holds numbers, and only numbers

# InvalidInputError: provider 'qdrant' cannot express lt on the non-numeric value 'm'.
#                    Qdrant can only order numbers. Use eq/in for strings.
```

Loud and unusable beats quiet and wrong. The failure lands on the developer who wrote
the filter, at the moment they write it, rather than on a customer months later.

## `ne` is not `Not(eq)`, and the difference is the records that are missing the field

Here is the corpus the conformance suite runs on. Note the last record: it has no
`tenant` at all, because in a real corpus something always doesn't.

| id | tenant | year |
| --- | --- | --- |
| `a` | acme | 2024 |
| `b` | acme | 2020 |
| `c` | globex | 2024 |
| `orphan` | — | — |

Now:

```python
ne("tenant", "acme")    # → {c}
~eq("tenant", "acme")   # → {c, orphan}
```

Both readings are defensible; what matters is that all three backends agree on one of
them, and that the one they agree on is safe. The rule:

- **A comparison against a field that is not there is false.** All of it — `eq`, `gt`,
  and yes, `ne`. So `ne("tenant", "acme")` means *"has a tenant, and it differs"*. The
  `orphan` does not have a tenant, so it does not satisfy a comparison about tenants.
- **`Not(f)` is a true logical negation.** `~eq("tenant", "acme")` asks *"is it not the
  case that this matches?"* — and it is indeed not the case for the `orphan`, which
  therefore matches.

The first rule is the load-bearing one. Its opposite — an absent field satisfying `ne` —
means a filter written to *exclude* one tenant silently admits every document that
forgot to say which tenant it belonged to. Which, again, is a fail-open.

`exists("tenant", present=False)` is how you ask for the `orphan` on purpose.

## The conformance suite, and the bug it caught

Each adapter compiles that algebra to something different, and here is the trap: a unit
test of a *compiler* can only check the compiler against the same understanding that
produced it. Misread what `ne` should do, and you will write the compiler wrong and then
write a test that agrees with it — and both will be wrong together, for years, in
agreement.

So the semantics are pinned **once, behaviourally**, in `test_conformance.py`. It asks
every question of the in-process store, whose implementation is thirty lines of Python
you can read in one sitting. Every other backend is then required to give the same
answers to the same questions.

It earned its keep immediately. Qdrant's natural spelling of `ne` is a bare `must_not`:

```python
Filter(must_not=[FieldCondition(key="tenant", match=MatchValue(value="acme"))])
```

This looks right. It reads right. It is what you would write. And it **also matches the
`orphan`** — because a record with no tenant is, indeed, not one whose tenant is acme.
Qdrant disagreed with the reference implementation on exactly the case that matters, in
the direction that fails open, and no test of the Qdrant compiler alone would ever have
said so. The fix is a presence guard in the same `must_not`:

```python
Filter(must_not=[
    FieldCondition(key="tenant", match=MatchValue(value="acme")),
    IsEmptyCondition(is_empty=PayloadField(key="tenant")),   # ← and it must have one
])
```

If you add a backend, you make it pass this suite. That is the whole contract.

## Two promises the port makes about search

**Higher is always better.** Cosine similarity ascends, Euclidean distance *descends*,
and dot product does whatever the vectors' magnitudes do. A caller that had to know
which metric a collection was built with in order to know which way to sort is a caller
that will one day sort backwards and return the *least* relevant documents — a bug that
looks exactly like "retrieval is bad" and is never traced to a comparison operator. So
every adapter normalises: `SearchHit.score` descends, always, whatever the metric. For
Euclidean that means `1 / (1 + d)`.

**The filter is applied before the cut, not after.** The most common bug in
hand-rolled vector search is: fetch the top *k*, then filter them, and return whatever
survives. It never raises. It just returns three results where five were asked for, and
sometimes zero — and it does not look like a bug, it looks like *"retrieval is a bit
weak for that customer"*, which is why it survives for years. Every adapter here pushes
the filter into the backend's own query, so `top_k=5` means five matching records if
five exist.

## What is different about each backend

### Qdrant: it does not accept our ids

Qdrant point ids must be an unsigned integer or a UUID. Ours are strings —
`"doc-42#chunk-3"`, a URL, whatever the document pipeline produced — and Qdrant will
reject them.

So the adapter mints one: `uuid5(namespace, record_id)`. Deterministic, so the same
document id always maps to the same point and `upsert` still overwrites rather than
duplicates. But a uuid5 is one-way, so the real id is also carried in the payload under
`__sarb_id`, and read back from there on the way out. Callers never see the uuid, and
nothing in the framework ever asks Qdrant to reproduce a string it cannot store.

### pgvector: there is no such thing as a collection

Qdrant has collections. Postgres has tables, and *that is our problem, not the user's* —
the port promised collections, so the adapter has to invent them.

Each collection becomes a table (`sarb_documents`), created on `ensure_collection`,
along with an HNSW index on the vector and a **GIN index on the metadata**. The second
one is not optional: without it every filtered search is a sequential scan of the JSONB
column, and the filter that was supposed to make the query fast is what makes it slow.

A registry table `sarb_collections` records each collection's dimensions and distance,
because Postgres will not tell you what a `vector(384)` column was meant for and
`ensure_collection` has to be able to refuse a reshape.

**A table name cannot be a bind parameter.** This is the one place in `sarb` where a
string from config is interpolated into SQL, and there is no `$1` that will save you —
Postgres does not parameterise identifiers. So the name is not escaped, it is
**validated**, against `^[a-z_][a-z0-9_]{0,48}$`, and a name that is not obviously safe
is not used at all. Everything else — every value, every filter operand — is a bind
parameter, always.

**Three-valued logic bites here and nowhere else.** In SQL, `metadata->>'tenant' <>
'acme'` is `NULL` when the key is absent, and `NULL` is not `TRUE`, so that predicate
already excludes the `orphan`... in that direction. But nesting it under a `NOT` flips
it into a different wrong answer. Rather than reason about where the `NULL`s land in
each combination, every comparison carries its own presence guard:

```sql
metadata ? $1 AND metadata->>$1 <> $2
```

Explicit, and the same shape for all eight operators. The conformance suite confirms
it lands where `memory.py` lands.

### `memory`: a real store that happens to be the default

`provider: memory` is not a stub or a mock. It does exact nearest-neighbour search over
every record it holds and a complete evaluation of the filter algebra — so an engine
tested against it is tested against real geometry, and a retrieval test that asserts an
ordering is asserting something true.

It is the default in `config/base.yaml`, which means a fresh checkout indexes and
retrieves with nothing installed and no key. It is also `O(n)` per query and forgets
everything on restart, which is fine for the thousands of vectors a developer has and
hopeless for the millions a production system has. That is a deliberate cliff: you
cannot point it at production by accident.

And it is the reference implementation. When Qdrant and `memory` disagree, `memory` is
right — not because it is better, but because it is the one whose semantics are legible
enough to argue about.

## What will bite you in production

**Filtered HNSW can under-return, and it does it quietly.** An HNSW index walks a graph.
A filtered search walks it looking for rows that *also* satisfy the filter — and if the
filter is selective (one tenant out of a thousand), the walk can exhaust its budget
before it finds `top_k` matches. You get fewer results than exist. Nothing raises.

This is a property of the index, not of this framework, and no adapter can hide it. What
the adapter can do is put the dial where you can reach it:

```yaml
vector_stores:
  default:
    provider: pgvector
    options:
      dsn: ${DATABASE_URL}
      collection: documents
      ef_search: 200 # default 64. Higher: more accurate, slower
```

If a highly-filtered search returns suspiciously few results, this is the first thing to
raise. The right value depends on your data, not on this framework, which is why there
is no clever default.

**`ensure_collection` is idempotent but will not reshape.** Calling it again with the
same shape does nothing. Calling it with a different `dimensions` or `distance` raises,
rather than silently writing 768-wide vectors into a 1536-wide collection — a model
change that was not carried through to the index corrupts retrieval in a way that
surfaces as *"the answers got worse"* weeks later and is never traced back.

**`delete` requires a selector.** There is no `delete(collection)` that means "all of
it", because that call is one slip of the wrist away from erasing a production index.
`ids=` or `where=`, exactly one, or `InvalidInputError`.

## API reference

::: sarb.providers.vectorstore
