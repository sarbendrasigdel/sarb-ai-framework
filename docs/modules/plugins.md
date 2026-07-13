# Plugin system

The registry answers one question: **which adapter is behind this config key?**

```yaml
llms:
  default:
    provider: anthropic     # <- a registry key
```

It holds factories, not instances. Nothing here opens a connection; that is the
[container's](dependency-injection.md) job.

## Writing a plugin

A plugin is an ordinary Python package that advertises a `register` function:

```toml
# acme-sarb/pyproject.toml
[project.entry-points."sarb.plugins"]
acme = "acme.sarb:register"
```

```python
# acme/sarb.py
from sarb.plugins import ComponentContext, ComponentRegistry
from sarb.config import parse_options


def register(registry: ComponentRegistry) -> None:
    registry.add("llms", "acme", build_acme, summary="ACME's hosted model")


def build_acme(ctx: ComponentContext) -> AcmeProvider:
    options = parse_options(ctx.config, AcmeOptions, component=ctx.path)
    return AcmeProvider(options)
```

Install it, and it is available:

```yaml
llms:
  default:
    provider: acme
```

That is the whole extension mechanism. A plugin can add a provider to an existing
family, or introduce an entirely new family — a reranker, a guardrail — reading
its config from `extras` without a single core file changing.

## The registry is handed to you; there is no global

Note the signature: `register(registry)`. The plugin **receives** the registry.

There is deliberately no module-level `REGISTRY` to import and mutate. No
import-time side effects, no ordering dependency between plugins, and a test can
build a throwaway registry that vanishes when the test ends:

```python
registry = ComponentRegistry()
registry.add("llms", "fake", lambda ctx: FakeLLM())
```

First-party adapters use exactly this mechanism. The only difference is that
`sarb.plugins.builtins.register` is called directly rather than discovered
through an entry point — which saves a metadata scan on every boot. There is no
privileged path: if the plugin API could not express the built-ins, that would be
a bug in the plugin API.

## Registration never imports the vendor SDK

This is the property that lets `pip install sarb` stay small without making the
missing-extra experience miserable.

`registry.add(...)` records a **factory**. The `import anthropic` lives *inside*
that factory and does not run until a component is actually built. So every
first-party adapter is registered whether or not its extra is installed, and
asking for one you have not installed produces:

```
Provider 'qdrant' requires an optional dependency: pip install 'sarb[qdrant]'
```

instead of a `ModuleNotFoundError` surfacing from somewhere in the import graph
of an unrelated script.

!!! danger "The one rule for adapter modules"
    Never import a vendor SDK at module level in anything the registry imports.
    CI enforces this: one job installs the core alone and imports it, and
    [`tests/architecture/`](../architecture/overview.md) fails if a vendor SDK
    appears outside its own adapter.

## Duplicate keys are an error

```
llms.anthropic is already registered by 'builtin'. Pass replace=True to override
it deliberately.
```

Two plugins silently claiming `anthropic` would make behaviour depend on
installation order — the kind of bug that takes a week to find. Overriding is a
legitimate thing to want, so it is allowed. It just has to be said out loud:

```python
registry.add("llms", "anthropic", my_patched_build, replace=True)
```

Built-ins register first, then plugins in name order. So a plugin *can*
deliberately replace a built-in, and cannot do so by accident.

## Loading policy

Loading a plugin means importing third-party code and calling it. That is a
capability, so it is governed:

```yaml
plugins:
  enabled: true
  allowlist: [acme] # exclusive: nothing else loads
  denylist: [globex] # applied on top
```

- `enabled: false` turns discovery off entirely.
- `allowlist`, when non-empty, is **exclusive**. A deployment that must not
  execute arbitrary installed code sets this — and then an attacker who manages
  to get a package *installed* still cannot get it *loaded*.
- A name in both lists is refused. That is the safe reading of a contradictory
  policy.

### A failing plugin stops the boot

A plugin that cannot be imported, does not resolve to a callable, or raises while
registering is a **hard error**:

```
PluginError: Plugin 'acme' failed while registering its components: ...
```

Booting "successfully" with half the providers missing, then failing on the first
request that happens to need one, is strictly worse than not booting. The
plugin's name is always in the message — with a dozen installed, "a plugin
failed" is not an actionable sentence.

## ComponentContext

What a factory gets, and nothing more:

| Field | What it is |
| --- | --- |
| `config` | this component's own `provider` + opaque `options` |
| `path` | `"llms.judge"` — pass to `parse_options` so errors name the YAML key |
| `family`, `name` | the two halves of that path |
| `settings` | full app settings, for cross-cutting values (timeouts, environment) |
| `resolver` | to resolve *other* components this one depends on |

`resolver` is the narrow `ComponentResolver` protocol, not the container: a
factory can reach its dependencies but cannot rebind or shut down the application
it lives in.

Reading *another component's* `options` through `settings` is a mistake — it
couples your adapter to that component's schema. Resolve the component instead.

## Inspecting what is registered

```python
registry.providers("llms")        # ('anthropic', 'gemini', 'ollama', 'openai')
registry.families()               # ('embedders', 'llms', 'vector_stores')
registry.supports("llms", "acme") # True
registry.lookup("llms", "acme").source  # 'acme'  — where did this come from?
```

`source` makes "why is this adapter here?" answerable at runtime, which matters
the first time a plugin overrides something and nobody remembers that it does.

## API reference

::: sarb.plugins
