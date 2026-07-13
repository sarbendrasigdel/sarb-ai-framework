# Your first contribution

A complete walkthrough: adding an LLM provider, from an empty file to a green CI run.

We will add **Mistral**. It is a good first task because it is genuinely useful,
genuinely small, and it touches every layer of the framework exactly once — so by the
end you will have seen the whole vertical slice with your own hands on it.

The goal is this, and nothing more:

```yaml
llms:
  default:
    provider: mistral
    options:
      model: mistral-large-latest
```

!!! note "Before you start"
    Read [`src/sarb/providers/llm/anthropic.py`](https://github.com/sarbendra/sarb/blob/main/src/sarb/providers/llm/anthropic.py)
    and its test file. You are about to write the same four pieces, and copying a
    working example beats inventing one.

## What you will touch

Five files. Note what is **not** on the list: nothing in `contracts/`, nothing in
`config/`, nothing in `di/`. If you find yourself editing one of those to add a
provider, stop — something has gone wrong, and it is worth understanding what before
you go further.

| File | Why |
| --- | --- |
| `pyproject.toml` | declare the `mistral` extra |
| `src/sarb/providers/llm/mistral.py` | the adapter |
| `src/sarb/providers/llm/__init__.py` | export it |
| `src/sarb/plugins/builtins.py` | map `"mistral"` → its factory |
| `tests/unit/providers/llm/test_mistral.py` | the tests |

## 1. The extra

```toml
# pyproject.toml
[project.optional-dependencies]
mistral = ["mistralai>=1.2"]
```

Add it to the `all` meta-extra too, and run `uv sync --all-extras --group dev`.

The core stays small on purpose: `uv add sarb` must never drag in an SDK the user did
not ask for. Everything vendor-shaped lives behind an extra.

## 2. The options schema

The adapter — not the config module — declares what its YAML looks like.

```python
# src/sarb/providers/llm/mistral.py
PROVIDER = "mistral"
EXTRA = "mistral"


class MistralOptions(LLMOptions):
    """Options for the Mistral adapter."""

    api_key: SecretStr | None = None
```

`LLMOptions` already gives you `model`, `base_url`, `timeout`, `max_retries` and the
token limits. Add only what Mistral genuinely needs beyond that — every option is a
thing a future reader has to understand, and a thing you have to keep working.

Two inherited behaviours you get for free, and should not undo:

- `model` has **no default**. A framework that picks a model for you picks the wrong
  one and bills you for it.
- `extra="forbid"`. A typo — `temprature: 0.7` — fails loudly instead of leaving an
  operator convinced they set something.

## 3. The adapter

```python
class MistralProvider:
    """The `LLMProvider` port, backed by Mistral's API."""

    def __init__(self, client: Mistral, options: MistralOptions) -> None:
        self._client = client
        self._options = options
```

**The client is injected. The adapter does not construct it.** This is the single most
important line on this page, and the reason is in the next section: it is what makes
the adapter testable without a key, a network, or a mock library.

Then implement the port — `generate`, `stream`, `capabilities`, `aclose` — and lean on
[`_shared.py`](../modules/llm-providers.md#what-the-adapters-share) for everything the
adapters have in common:

| Use | For |
| --- | --- |
| `ResponseBuilder` | assembling a streamed response, including parallel tool calls |
| `parse_arguments` | tool arguments, including the model's malformed JSON |
| `error_for_status` | mapping HTTP status to `sarb.errors` with the right retry semantics |
| `unsupported(...)` | anything the vendor's API cannot express |

That last one is a rule, not a helper:

!!! danger "An adapter that cannot express a request raises. It never silently drops it."
    If Mistral has no `tool_choice: required`, then `ToolChoice.REQUIRED` raises
    `InvalidInputError` naming the provider and the feature. Quietly sending `auto`
    instead means the caller discovers the difference from a wrong answer, or a bill.

## 4. The factory

```python
def build(ctx: ComponentContext) -> MistralProvider:
    """Construct the adapter from `llms.<name>` config."""
    sdk = import_optional("mistralai", extra=EXTRA, used_by=f"LLM provider {PROVIDER!r}")
    options = parse_options(ctx.config, MistralOptions, component=ctx.path)
    return MistralProvider(sdk.Mistral(api_key=..., ...), options)
```

Three things are happening, and each has a reason:

- **`import_optional`, not `import mistralai`, and inside the function.** A top-level
  vendor import runs on every boot — `plugins/builtins.py` imports all adapter modules
  — so it would make the extra mandatory for everyone. Instead, a missing extra
  produces `MissingDependencyError: ... uv add "sarb[mistral]"`, which is an
  instruction rather than a traceback.
- **`parse_options(..., component=ctx.path)`** so a bad value names the YAML key the
  operator actually typed: `llms.default.options.timeout: Input should be greater than 0`.
- **The client is constructed *here*** and handed to the adapter, which is what makes
  the adapter testable.

## 5. Register it

```python
# src/sarb/plugins/builtins.py
registry.add(
    "llms", "mistral", mistral.build,
    summary="Mistral's hosted models. Needs sarb[mistral].",
)
```

Register it **unconditionally** — installed or not. Registering only when the SDK is
importable would make `sarb providers list` differ per machine, and would turn a
missing extra into "no such provider", which sends the user hunting for a typo instead
of running one command.

At this point `provider: mistral` works. Now make it *stay* working.

## 6. The tests

Two rules, and they are the difference between tests that catch bugs and tests that
launder them.

**Build fixtures from the vendor's own models.**

```python
from mistralai.models import ChatCompletionResponse   # not a hand-written dict
```

A dict you wrote by hand can be wrong in exactly the same way your adapter is wrong,
and then the two agree with each other, happily, forever, while production fails.
Constructing the SDK's own Pydantic model makes the SDK validate the shape you are
*claiming* to receive.

**Inject a recording client, and assert on what would have gone over the wire.**

```python
llm = MistralProvider(FakeClient(reply(text("Paris."))), options)

response = await llm.generate(LLMRequest(messages=(Message.user("hi"),), system="Be terse."))

assert response.text == "Paris."
assert sent(llm)["system"] == "Be terse."     # ← the assertion that catches real bugs
```

The second assertion is the one that matters. Translation bugs are silent: a test that
only checks the answer passes cheerfully while the system prompt is being dropped on
the floor.

Cover, at minimum: the system prompt, a tool call round-trip, a streamed response
ending in an assembled `StreamEnd`, malformed tool arguments, and each error class
(`401` → `AuthenticationError`, `429` → `RateLimitError` with `retry_after`, `5xx` →
retryable). The four existing adapter test files are your template — copy the structure.

## 7. Documentation

Add a row to the table in [LLM providers](../modules/llm-providers.md), and if Mistral
has a quirk worth knowing — a role it renames, an id it does not issue — write it in
the "Where the vendors actually differ" section. That section is the accumulated
scar tissue of this module, and it only stays useful if people keep adding to it.

## 8. Ship it

```bash
uv run ruff check --fix . && uv run ruff format .
uv run mypy
uv run pytest
uv run --group docs mkdocs build --strict
```

All four green, then open the PR. See [conventions](conventions.md) for what reviewers
will look at, and [testing](testing.md) for the architecture tests that will fail you
if the vendor SDK escaped its adapter.

## The same shape, everywhere else

You have now written an LLM adapter, but nothing above was specific to LLMs. An
embedding provider, a vector store, a document loader, a storage backend: same four
pieces, same rules, different port.

```
options schema  →  adapter (client injected)  →  build() (lazy import)  →  registry key
```

Learn it once here, and the rest of the framework is a variation on it.
