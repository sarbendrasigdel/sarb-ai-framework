# Configuration

Every other module reads its settings from here, so this module has one job:
**make a misconfigured process fail at startup, loudly, with a message that says
what to change** — rather than at 3am, on the first request that happens to touch
the broken path.

```python
from sarb.config import load_settings

settings = load_settings(config_dir="config", profile="production")
llm = settings.component("llms", "default")
```

## Layering

Values are resolved from six layers. Highest priority first:

| # | Layer | Set by | Typical use |
| --- | --- | --- | --- |
| 1 | `overrides=` | the caller | tests; embedding `sarb` in a larger app |
| 2 | Environment | `SARB_*` variables | per-deployment values, secrets |
| 3 | `.env` file | `env_file=` | local development |
| 4 | YAML | `base.yaml` + `<profile>.yaml` | everything structural |
| 5 | Secret files | `/run/secrets/*` | Docker / Kubernetes secrets |
| 6 | Defaults | field defaults | sensible baseline |

Layers **deep-merge**, they do not replace. This is the property that lets one
container image run in every environment:

```bash
# Overrides exactly one leaf. The rest of the YAML block survives.
SARB_LLMS__DEFAULT__OPTIONS__TEMPERATURE=0.9
```

`__` is the nesting delimiter, `SARB_` the prefix. So
`SARB_LLMS__DEFAULT__OPTIONS__MODEL` reaches `settings.llms["default"].options["model"]`
without disturbing the sibling keys next to it.

## Profiles

```
config/
  base.yaml          # true everywhere
  development.yaml   # overlays
  staging.yaml
  production.yaml
```

`base.yaml` is merged, then `<profile>.yaml` on top. The profile comes from
`load_settings(profile=...)` or `$SARB_PROFILE`.

A **missing profile file is an error**, not a fallback. Asking for `production`
and silently getting development defaults is precisely the failure mode this
module exists to prevent.

## Secrets

YAML files are committed; secrets are not. So YAML *references* secrets and never
contains them:

```yaml
llms:
  default:
    provider: anthropic
    options:
      api_key: ${ANTHROPIC_API_KEY}          # required — load fails if unset
      base_url: ${ANTHROPIC_BASE_URL:-https://api.anthropic.com}   # optional
```

- `${VAR}` — required. If unset, `load_settings()` raises immediately: a
  container missing its API key dies at boot with a precise message, instead of
  serving traffic until the first model call.
- `${VAR:-default}` — optional, with a fallback.
- `$${VAR}` — a literal `${VAR}`. Prompt templates legitimately contain `${...}`
  and must survive interpolation intact.

### Redaction

`settings.redacted()` masks every `SecretStr` and every option key that *looks*
like a credential (`api_key`, `apiKey`, `auth_token`, `password`, `client_secret`,
`dsn`, …; matching is case- and style-insensitive).

Log it at startup. Most production incidents that trace back to configuration are
diagnosed in seconds when the effective config was logged, and in hours when it
was not.

```python
logger.info("configuration loaded", extra=settings.redacted())
```

## Components

A component is one instance of a replaceable thing — an LLM, a vector store, a
memory backend:

```yaml
llms:
  default:
    provider: anthropic       # registry key of an adapter
    enabled: true
    options:                  # opaque to the kernel; the adapter validates it
      model: claude-sonnet-5
```

Families are **maps**, because production runs several models at once:

```yaml
llms:
  default: { provider: anthropic, options: { model: claude-sonnet-5 } }
  fast:    { provider: anthropic, options: { model: claude-haiku-4-5 } }
  judge:   { provider: openai,    options: { model: gpt-4o } }

vector_stores:
  default: { provider: qdrant, options: { url: ${QDRANT_URL}, collection: docs } }
```

```python
settings.component("llms")             # the "default" one
settings.component("llms", "judge")    # by name
settings.enabled_components("llms")    # every enabled member
```

Ask for a name that is not configured and the error tells you what *is*:

```
No component named 'judge' in family 'llms'. Available: default, fast
```

Built-in families: `llms`, `embedders`, `vector_stores`, `memories`, `storages`,
`prompts`, `tools`. A plugin that introduces a new family (say, rerankers) reads
its config from `extras` — without the kernel growing a field for it:

```yaml
extras:
  rerankers:
    default: { provider: cohere }
```

## Adapter options

The kernel treats `options` as an opaque dict, on purpose — see
[the architecture overview](../architecture/overview.md#2-the-kernel-never-learns-a-vendors-name)
for why. The typing happens in the adapter, which is the only place that knows
what its vendor needs:

```python
from pydantic import BaseModel, SecretStr
from sarb.config import ComponentConfig, parse_options


class AnthropicOptions(BaseModel):
    """Options for the Anthropic adapter. The kernel never sees this class."""

    model: str = "claude-sonnet-5"
    api_key: SecretStr
    max_tokens: int = 4096
    timeout_s: float = 60.0


class AnthropicProvider:
    def __init__(self, config: ComponentConfig) -> None:
        self._options = parse_options(config, AnthropicOptions, component="llms.default")
```

`parse_options` reports failures in the operator's vocabulary — the config path,
not the Python field — and reports all of them at once:

```
Invalid options for provider 'anthropic':
  llms.default.options.api_key: Field required;
  llms.default.options.max_tokens: Input should be a valid integer
```

## Guardrails

Some misconfigurations are refused outright, at load:

| Rejected | Because |
| --- | --- |
| `app.debug` with `app.environment: production` | debug mode leaks internal errors to clients |
| `telemetry.enabled` with no `telemetry.endpoint` | telemetry with nowhere to go fails silently |
| `database.pool_min_size > pool_max_size` | the pool deadlocks on first use |
| any unknown key | a typo in config must never be silently ignored |

Validation errors name the field path *and* the environment variable that sets it,
because the person reading the log is usually not the person who wrote the model:

```
Invalid configuration (1 problem(s)):
  - server.port: Input should be less than or equal to 65535  [env: SARB_SERVER__PORT]
  config_dir=config profile=production
```

## Immutability, and the absence of a global

`Settings` is **frozen**. A running process does not mutate its own
configuration; reconfiguring means building a new object and installing it at a
defined point, which keeps config changes observable rather than ambient.

There is deliberately **no module-level `settings = Settings()`**. Evaluating
settings at import time makes every importing module unusable without a valid
environment — the single most common way a config layer becomes untestable.

Prefer injection. Where injection is genuinely impractical, there is a
context-scoped ambient accessor:

```python
from sarb.config import current_settings, load_settings, use_settings

with use_settings(load_settings(config_dir="config")):
    ...                       # current_settings() works in here
    current_settings().app.name
```

Because it is backed by a `ContextVar`, a test can swap the settings for the
duration of a block without leaking into neighbouring tests or concurrent tasks.

## Testing

`load_settings()` is pure: same inputs, same output, no global state. Tests pass
`overrides=` and never touch the filesystem or the environment:

```python
settings = load_settings(
    overrides={"llms": {"default": {"provider": "fake"}}},
)
```

For YAML interpolation, inject the environment rather than mutating the real one:

```python
settings = load_settings(config_dir=tmp_path, environ={"ANTHROPIC_API_KEY": "sk-test"})
```

## API reference

::: sarb.config
