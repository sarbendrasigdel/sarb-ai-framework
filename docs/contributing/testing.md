# Testing

Three kinds of test live in this repo, and they exist for three different reasons.

| Where | What it protects |
| --- | --- |
| `tests/unit/` | behaviour — does this module do what it says? |
| `tests/architecture/` | the design — is the dependency rule still true? |
| `tests/unit/providers/test_wiring.py` | the promise — does YAML still become a working provider? |

Run them:

```bash
uv run pytest                       # everything, ~12 seconds
uv run pytest tests/architecture -q # the design, on its own
uv run pytest --cov --cov-report=term-missing
```

`asyncio_mode = "auto"` is set, so an `async def test_...` needs no decorator.

## The architecture tests

These are the ones that will fail your PR in a way that feels unfair the first time.
They are worth understanding rather than working around.

`tests/architecture/test_dependency_rule.py` parses every module under `src/sarb` with
`ast` — no imports are executed, so it runs with no extras installed — and asserts two
invariants.

### 1. The dependency rule

```python
FORBIDDEN_IMPORTS = {
    "contracts":  {"engines", "providers", "api", "cli", "di", "plugins", "config"},
    "engines":    {"providers", "api", "cli"},
    "providers":  {"engines", "api", "cli"},
    ...
}
```

Read each row as: *code in `<key>` must not know that `<values>` exist.* The important
one is `engines` → `providers`. If a retrieval engine imports a Qdrant adapter, Qdrant
is no longer replaceable — and the import that does it is always one plausible-looking
line in an otherwise-good PR. Review does not reliably catch that. A test does.

### 2. Vendor containment

```python
VENDOR_CONTAINMENT = {
    "anthropic": "sarb.providers",
    "qdrant_client": "sarb.providers",
    "fastapi": "sarb.api",
    "opentelemetry": "sarb.observability",
    ...
}
```

A vendor SDK may only be imported inside the subpackage that adapts it. This is what
keeps `uv add sarb` small, and what guarantees that swapping a vendor touches one
directory. A separate CI job (`core-install`) installs the core alone and imports it,
which catches the same mistake from the other direction.

Adding a new provider means adding its SDK to this map. That is the intended workflow —
the map is a statement of intent, not an obstacle.

### 3. The rules cover the package

If someone adds `src/sarb/rerankers/` and no rule for it, the parametrised tests above
would keep passing while checking nothing at all. So a third test asserts every
subpackage has an entry. An empty `frozenset()` is a valid, deliberate answer; silence
is not.

## Testing an adapter

Adapters are 90% translation — our models into a vendor's request, a vendor's response
back into our models — and translation is where the bugs live. Two rules make those
bugs findable.

**Build fixtures from the vendor's own response models.**

```python
anthropic = pytest.importorskip("anthropic")

from anthropic.types import Message as AnthropicMessage, TextBlock, Usage


def reply(*content, stop_reason="end_turn") -> AnthropicMessage:
    return AnthropicMessage(
        id="msg_1", model="claude-opus-4-8", role="assistant", type="message",
        stop_reason=stop_reason, content=list(content),
        usage=Usage(input_tokens=10, output_tokens=5),
    )
```

A hand-written dict can be wrong in exactly the same way the adapter is wrong, and the
two will then agree with each other forever while production fails. Constructing the
SDK's own model makes the SDK validate the shape we claim to receive.

(This is why `pytest.importorskip` sits above the imports, and why those files have an
`E402` exemption in `pyproject.toml`. It is a deliberate exception, and it is written
down as one.)

**Inject a recording client and assert on the outbound request.**

```python
def sent(llm):
    """The kwargs the adapter would actually have sent."""
    return llm._client.messages.params


assert sent(llm)["system"] == "Be terse."
```

Checking only the returned text passes happily while the system prompt is dropped on
the floor. The outbound assertion is the one that catches real bugs.

Ollama's tests go one step further and use a real `httpx.MockTransport`, so the request
being asserted is byte-for-byte the one that would have gone on the wire.

## Testing an engine

Engines are tested against **the fake provider**, which is a real registered provider —
not a mock:

```python
llm = FakeProvider(FakeOptions(
    responses=("Paris.",),
    tool_calls=(ScriptedToolCall(name="search", arguments={"q": "capital"}),),
))
```

It can be scripted to call a tool, to emit malformed JSON, to be slow, or to fail with
any error class the framework defines — which is the only honest way to test a retry
policy. The alternative is getting rate limited by a real vendor, deliberately, in CI.

It also records what it was sent:

```python
assert llm.requests[0].system == "Be terse."
```

Use that. Prompt-assembly bugs produce plausible answers and silent regressions.

An engine test that needs a *specific vendor* to pass is a bug in the engine — it means
the engine has learned something about a provider it is not supposed to know.

## What a good test looks like here

Look at any file in `tests/unit/providers/llm/`. The pattern:

- **The name is the claim.** `test_a_tool_turn_becomes_tool_result_blocks_in_one_user_message`,
  not `test_tools_2`. When it fails at 3am, the name is the whole bug report.
- **A comment says why the test exists**, when that is not obvious — usually naming the
  bug it prevents. `# Losing the signature means the next turn is rejected by the API.`
- **One behaviour per test.** A test asserting six things fails on the first and hides
  the other five.
- **No mock library.** Ports are Protocols; a plain class with the right shape *is* a
  provider. If you find yourself reaching for `unittest.mock`, the design is probably
  telling you something.

## Before you push

```bash
uv run ruff check --fix . && uv run ruff format .
uv run mypy                                  # strict, and it means it
uv run pytest
uv run --group docs mkdocs build --strict
```

That is exactly what CI runs. Nothing is hidden, and there is no reason to discover a
failure from a red build.
