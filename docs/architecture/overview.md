# Architecture overview

`sarb` is a framework for building AI products — RAG systems, chatbots,
assistants, document intelligence, workflow agents — that survive the vendors
they were built on.

That last clause is the whole design brief. The models, the vector databases and
the embedding runtimes in use today will not be the ones in use in three years.
A framework whose business logic names them will have to be rewritten when they
change. So the organising question behind every decision here is: **what has to
be true for a vendor swap to be a config change rather than a project?**

## The rings

Dependencies point inward. Nothing else about the layout matters as much.

```
┌──────────────────────────────────────────────────────────────┐
│  api/   cli/                        delivery                 │
│  FastAPI routers, streaming, CLI commands                    │
├──────────────────────────────────────────────────────────────┤
│  engines/                           use-cases                │
│  agent, retrieval, memory, workflow, documents, tools        │
├──────────────────────────────────────────────────────────────┤
│  contracts/                         ports + models           │
│  LLMProvider, VectorStore, Message, Document, Chunk …        │
└──────────────────────────────────────────────────────────────┘
              ▲                              ▲
              │ implements                   │ depends on
┌─────────────┴──────────────┐  ┌────────────┴─────────────────┐
│  providers/     adapters   │  │  config/ di/ observability/  │
│  anthropic, openai, qdrant │  │  plugins/       the kernel   │
└────────────────────────────┘  └──────────────────────────────┘
```

| Ring | May import | Never imports |
| --- | --- | --- |
| `contracts` | pydantic, stdlib | anything else in `sarb` |
| `engines` | `contracts`, kernel | `providers`, `api`, `cli` |
| `providers` | `contracts`, kernel, **one** vendor SDK | `engines`, `api`, `cli` |
| `middleware` | `contracts`, kernel | `providers`, `engines`, `api`, `cli` |
| `api`, `cli` | `engines`, `contracts`, kernel | — |
| kernel | `contracts` | `engines`, `providers` |

**The rule that carries the design: `engines/` may never import `providers/`.**

A retrieval engine knows the `VectorStore` port. It does not know Qdrant exists.
The day someone imports `sarb.providers.vectorstores.qdrant` into an engine "just
for a type hint", Qdrant stops being replaceable — and it will look like a
perfectly reasonable one-line diff at review time.

So it is not a convention. It is a test — `tests/architecture/test_dependency_rule.py`
parses every module and fails CI on violation. Conventions decay; tests do not.

## Five decisions, and why

### 1. Ports are `typing.Protocol`, not abstract base classes

```python
@runtime_checkable
class LLMProvider(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...
```

An ABC demands inheritance: to be an `LLMProvider` you must import `sarb` and
subclass it. A `Protocol` is structural: any class with the right shape *is* one.

This matters more than it looks. It means an adapter can be written with **zero
`sarb` imports**, that a downstream team can satisfy a port with a class they
already have, and that tests can substitute a plain object without a mock
library. It is the difference between "depend on our abstraction" and "depend on
a shape" — and only the second one genuinely minimises lock-in.

Composition over inheritance, applied to the type system itself.

### 2. The kernel never learns a vendor's name

The tempting way to type provider config is a discriminated union:

```python
llm: OpenAIConfig | AnthropicConfig | OllamaConfig   # do not do this
```

That union lives in the core. Every new provider — including one written by a
downstream team, or a third-party plugin — means editing a core file. The kernel
slowly accumulates knowledge of every vendor that has ever existed, and the
plugin system becomes a lie.

Instead, a component is an envelope:

```yaml
llms:
  default:
    provider: anthropic          # a registry key, resolved at wiring time
    options:                     # opaque to the kernel
      model: claude-sonnet-5
      api_key: ${ANTHROPIC_API_KEY}
```

The kernel validates the envelope. The **adapter** declares a Pydantic model for
its own options and validates its own slice at construction, via
[`parse_options`](../modules/configuration.md#adapter-options). Strong typing
exactly where the knowledge lives; a kernel that stays ignorant.

### 3. Components are named families, not one-of-each

Production systems do not run one model. They run a cheap one for routing, an
expensive one for synthesis, and a third as an evaluation judge — often from
different vendors, on different latency budgets.

So every family is a mapping, and the container resolves by name:

```yaml
llms:
  default: { provider: anthropic, options: { model: claude-sonnet-5 } }
  fast:    { provider: anthropic, options: { model: claude-haiku-4-5 } }
  judge:   { provider: openai,    options: { model: gpt-4o } }
```

```python
llm = container.get(LLMProvider)            # "default"
judge = container.get(LLMProvider, "judge")
```

Designing for one-of-each is the mistake that gets discovered in month four, when
it is load-bearing everywhere.

### 4. Async-first, with a sync façade

Every port is `async def`. Streaming tokens, calling three tools concurrently,
and serving FastAPI are all async-native, and async cannot be retrofitted onto a
sync core without breaking every interface in the framework — whereas a sync
façade over an async core is a thin, mechanical wrapper.

Async is the substrate; sync is the convenience.

### 5. A small core, with vendors behind extras

```bash
uv add sarb                            # pydantic, pyyaml, anyio. That is all.
uv add "sarb[anthropic,qdrant,api]"    # what you actually use
```

Vendor SDKs are imported **lazily, inside the adapter that needs them**, so a
missing extra fails at component construction with an actionable message rather
than at import with a `ModuleNotFoundError` in someone's unrelated script.

CI enforces this with a job that installs the core alone and imports it. A
framework whose core secretly requires a vendor SDK is not a framework.

## How a request flows

Nothing below the delivery layer knows which vendors were chosen:

```
HTTP request
   → api/            validates the request, resolves the container scope
   → engines/agent   plans, calls tools, loops  — sees LLMProvider, ToolProvider
   → contracts/      LLMRequest, ToolCall, Message
   → providers/      AnthropicProvider translates to and from the vendor's API
   → vendor SDK
```

Swapping Anthropic for OpenAI changes one line of YAML. Nothing above
`providers/` recompiles, and no test above `providers/` changes.

## Wiring, and the composition root

The rings are joined in exactly one place: the DI container, at startup.

```
config          → what to build          (llms.default = anthropic)
plugin registry → how to build it        ("anthropic" → AnthropicProvider)
DI container    → builds it, once, and hands it out by port
```

Wiring in one place is what allows every other module to stay ignorant. Any
module that constructs its own provider has quietly moved the composition root
into itself, and taken its own replaceability with it.

In practice that is two lines:

```python
from sarb.di import bootstrap

async with bootstrap(config_dir="config", profile="production") as container:
    llm = await container.get(LLMS)
```

See [Dependency injection](../modules/dependency-injection.md) for scopes and
lifecycle, and [the plugin system](../modules/plugins.md) for how the registry is
populated.

## Module map

| Ring | Module | Status |
| --- | --- | --- |
| kernel | [Configuration](../modules/configuration.md) | ✅ |
| kernel | [Dependency injection](../modules/dependency-injection.md) | ✅ |
| kernel | [Plugin registry](../modules/plugins.md) | ✅ |
| kernel | [Observability](../modules/observability.md) | ✅ |
| contracts | [Models + ports](ports.md) | ✅ |
| providers | [LLMs](../modules/llm-providers.md) — Anthropic, OpenAI, Gemini, Ollama | ✅ |
| providers | [Embeddings](../modules/embeddings.md) — OpenAI, FastEmbed, Sentence Transformers | ✅ |
| middleware | [Retry, limits, telemetry](../modules/middleware.md) | ✅ |
| providers | [Vector stores — Qdrant, pgvector, memory](../modules/vector-stores.md) | ✅ |
| providers | Storage — Postgres, Redis, object storage | ⬜ |
| engines | Prompt management | ⬜ |
| engines | Document processing (loaders, chunking) | ⬜ |
| engines | Retrieval | ⬜ |
| engines | Memory | ⬜ |
| engines | Tool calling | ⬜ |
| engines | Agent | ⬜ |
| engines | Workflow | ⬜ |
| engines | Evaluation | ⬜ |
| delivery | Authentication | ⬜ |
| delivery | REST + streaming API | ⬜ |
| delivery | CLI | ⬜ |

## What this architecture costs

Honesty about the trade-off, so the next maintainer can decide whether it still
holds:

- **Indirection.** Reading "how does a chat call actually reach Anthropic" means
  following a port to a registry to an adapter. That is a real cost, paid on
  every code-reading session, in exchange for a vendor swap being a config change.
- **Ceremony for one-off code.** A script that just wants to call one model does
  not need any of this. Such a script should use the vendor SDK directly. This
  framework earns its keep when the same infrastructure serves many products —
  which is exactly the premise it was commissioned under.
- **A slower first module.** The ports must exist before the adapters, and the
  kernel before the ports. The payoff arrives at product number two.
