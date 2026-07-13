# Dependency injection

The container is the one place where the rings are joined — where `llms.default`
stops being a string in a YAML file and becomes an `AnthropicProvider` holding a
real HTTP pool.

Every other module in this framework is written against ports and knows nothing
about the adapters behind them. That ignorance is only affordable because
*something* eventually does the wiring, and it matters enormously that it is
**exactly one something**. A module that constructs its own provider has quietly
moved the composition root into itself, and taken its own replaceability with it.

```python
from sarb.di import bootstrap

async with bootstrap(config_dir="config", profile="production") as container:
    llm = await container.get(LLMS)              # settings.llms.default
    judge = await container.get(LLMS, "judge")   # settings.llms.judge
```

## What the container resolves

**Configured components**, by family and name. *What* to build comes from
settings; *how* to build it comes from the registry:

```yaml
llms:
  default: { provider: anthropic, options: { model: claude-sonnet-5 } }
  judge:   { provider: openai,    options: { model: gpt-4o } }
```

```python
llm = await container.get(LLMS)             # -> LLMProvider, inferred
judge = await container.get(LLMS, "judge")  # -> LLMProvider, inferred
```

Resolution is type-safe end to end because a
[`Family`](../architecture/overview.md) is generic in its port:

```python
LLMS = Family("llms", LLMProvider)
```

**Bound services**, by type. Anything the composition root wires by hand:

```python
container.bind(Clock, factory=lambda c: SystemClock())
container.bind(Settings, instance=settings)   # already done for you

clock = await container.get(Clock)
```

## Scopes

| Scope | One per | Closed by the container | Use for |
| --- | --- | --- | --- |
| `SINGLETON` | application | ✅ at shutdown | anything with a connection pool: LLM clients, vector stores, databases |
| `SCOPED` | scope (≈ request) | ✅ on scope exit | per-request state: a unit of work, a conversation buffer, a request-bound session |
| `TRANSIENT` | every resolve | ❌ **never** | cheap, resource-free objects only |

```python
async with container.scope() as request:
    memory = await request.get(MEMORIES)   # built here
# `memory` is closed here. Singletons are untouched.
```

!!! warning "Transients are not tracked"
    A `TRANSIENT` component that owns a socket **will leak it** — the container
    never sees it again and never closes it. If a component holds a resource,
    make it `SCOPED`.

### Request scope is not silently promoted

Resolving a `SCOPED` component from the root container **raises**:

```
llms.session (postgres) is request-scoped and cannot be resolved from the
application container. Open a scope first: `async with container.scope() as request:`
```

The alternative — quietly handing back a singleton — is the bug where one user's
request-bound session gets captured by a long-lived object and served to the
next user. It is worth an error message.

## Lifecycle

Any component with an `aclose` method is closed automatically. No registration,
no base class: `Closable` is a structural protocol, so a vendor client that
already has `aclose` satisfies it with no shim.

Shutdown is **LIFO** — reverse creation order, because a component built later
may depend on one built earlier. Every component is closed **even if an earlier
one raises**, and the failures surface together as an `ExceptionGroup`. A
shutdown that silently drops errors is how connection leaks reach production.

## Writing a factory

A factory receives a [`ComponentContext`](plugins.md#componentcontext) and returns
the component. It may be sync or async — async if it has to open a connection,
rather than blocking the event loop inside `__init__`.

```python
from pydantic import BaseModel, SecretStr
from sarb.config import parse_options
from sarb.plugins import ComponentContext


class AnthropicOptions(BaseModel):
    model: str = "claude-sonnet-5"
    api_key: SecretStr
    max_tokens: int = 4096


def build(ctx: ComponentContext) -> AnthropicProvider:
    options = parse_options(ctx.config, AnthropicOptions, component=ctx.path)
    return AnthropicProvider(options)
```

Note what the adapter class itself takes: a plain typed options object. Not the
context, not the container, not the settings. It stays constructible — and
testable — without any of the framework around it.

A factory that needs *another* component reaches for the resolver:

```python
async def build(ctx: ComponentContext) -> Retriever:
    store = await ctx.resolver.resolve(VECTOR_STORES)
    embedder = await ctx.resolver.resolve(EMBEDDERS)
    return HybridRetriever(store, embedder)
```

`ctx.resolver` is deliberately the narrow `ComponentResolver` protocol, not the
container: a factory can resolve its dependencies but cannot rebind or shut down
the application it lives in.

## Design decisions

### No auto-wiring

The container will **not** introspect `__init__` annotations and construct
dependencies by magic. Auto-wiring is delightful in a demo and awful in year
three: it fails at runtime, in a stack trace nobody can read, usually during a
deploy. An explicit factory costs three lines and stays debuggable forever.

### Singletons are race-free

Ten requests arriving simultaneously on a cold container produce **one**
instance, not ten. Construction is guarded per key, so two *different* singletons
still build concurrently. A second HTTP pool created because two requests landed
at once is the kind of bug that only ever appears under load, in production.

### Cycles are reported, not deadlocked

```
Circular dependency: llms.default (a) -> llms.b (b) -> llms.default (a)
```

Detection happens *before* the per-key lock is taken — otherwise a cycle would
block on a lock the same task already holds, and the process would hang with no
message at all.

### Adapters are verified against their port at wiring time

When a port is `runtime_checkable`, a freshly built component is checked against
it:

```
Provider 'anthropic' built for 'llms.default' does not satisfy LLMProvider:
AnthropicProvider is missing one or more of its methods.
```

This only confirms the methods exist, not their signatures — but it catches an
adapter that has drifted out of shape at startup, rather than three layers deep
on the first call in production.

## Testing

`bind_component` swaps a fake into the container. The wiring above it — engines,
routers, the agent loop — runs for real, with no config, no credentials, and no
network:

```python
container = Container(load_settings(), registry)
container.bind_component(LLMS, FakeLLM())

result = await my_agent.run("hello")   # exercises the real code path
```

The substituted object is **not** closed by the container: the test made it, the
test owns it.

## API reference

::: sarb.di

::: sarb.contracts.component
