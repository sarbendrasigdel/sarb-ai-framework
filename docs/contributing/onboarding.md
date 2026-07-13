# Onboarding

This page is written for someone who has never seen this codebase. Read it once,
end to end, before you open a file. It should take about twenty minutes, and it will
save you considerably more than that.

## What this project is

`sarb` is a **framework**, not an application. Nothing here answers a question, ranks
a document, or serves a customer. It is the layer that products are built *on*, so
that the tenth AI product at this company does not rebuild the infrastructure the
first nine already had.

The single design brief, which explains almost every decision you will meet:

> The models, vector databases and embedding runtimes in use today are not the ones
> that will be in use in three years. **A vendor swap must be a config change, not a
> project.**

If a change you are about to make would make that less true, it is the wrong change,
however convenient it looks in isolation.

## The mental model in five minutes

There are three kinds of code in `src/sarb/`, and knowing which kind you are editing
tells you almost everything about what you are allowed to do.

```
contracts/     What things ARE.       LLMProvider, Message, Document, Chunk.
               Zero sarb imports. Zero vendor imports. Pure vocabulary.

engines/       What the system DOES.  Retrieval, agents, memory, workflows.
               Depends on contracts. Never learns that Qdrant exists.

providers/     How the outside world  AnthropicProvider, QdrantStore.
               is reached.            One vendor SDK each, imported lazily.
```

Plus a **kernel** (`config/`, `di/`, `plugins/`, `observability/`) that everything
depends on and which depends on nothing, and a **delivery** layer (`api/`, `cli/`)
that sits on top.

Arrows point inward. `engines/` sees a *port*; `providers/` implements it; the two
never meet in an import statement. They are introduced at runtime, in exactly one
place, by the DI container.

```
config          says WHAT to build      llms.default = anthropic
plugin registry says HOW to build it    "anthropic" → AnthropicProvider
DI container    builds it, once, and hands it out by port
```

That is the whole architecture. [The overview](../architecture/overview.md) explains
why each piece is shaped the way it is; come back to it once the code below makes
sense.

### The rule that carries the design

**`engines/` may never import `providers/`.**

Everything else is negotiable. This is not. The day a retrieval engine imports
`sarb.providers.vectorstores.qdrant` — "just for a type hint" — Qdrant stops being
replaceable, and the diff that did it looked entirely reasonable at review time.

Which is why it is not a convention but a test. `tests/architecture/` parses every
module in the package and fails CI on violation. See
[testing](testing.md#the-architecture-tests).

## Reading the code in the right order

Do this in one sitting. Each file assumes the one above it.

1. **`src/sarb/contracts/llm.py`** — the `LLMProvider` port and the models around it
   (`LLMRequest`, `Message`, `ToolCall`, `StreamEvent`). This is the vocabulary the
   whole framework speaks. Notice it is a `Protocol`, and that it names no vendor.
2. **`src/sarb/config/`** — how `config/base.yaml` becomes a `Settings` object, and
   why a component's `options` are an opaque `dict` the kernel refuses to look inside.
3. **`src/sarb/plugins/registry.py`** — a mapping from `"anthropic"` to a *factory*.
   Nothing is constructed here.
4. **`src/sarb/di/container.py`** — where the factory is finally called, once, and the
   result handed out by port.
5. **`src/sarb/providers/llm/anthropic.py`** — an adapter. Now you have seen a full
   vertical slice: YAML → registry → container → port → vendor.
6. **`tests/unit/providers/test_wiring.py`** — that same slice, asserted. If you read
   only one test, read this one.

By the end of (6) you should be able to answer: *what actually happens when I change
`provider: anthropic` to `provider: openai`?* If you cannot, ask — the answer is the
job.

## The local dev loop

```bash
uv sync --all-extras --group dev   # once
```

Then, while working:

```bash
uv run pytest                      # the suite. ~12 seconds; run it constantly
uv run pytest tests/unit/providers -q     # just what you are touching
uv run ruff check --fix . && uv run ruff format .
uv run mypy                        # strict. Not optional; CI runs it too
```

Docs are part of the change, not a follow-up:

```bash
uv run --group docs mkdocs serve   # http://127.0.0.1:8000
uv run --group docs mkdocs build --strict   # what CI runs. A dead link fails it
```

Everything CI does, you can run locally, and it is all in
[`.github/workflows/ci.yml`](https://github.com/sarbendra/sarb/blob/main/.github/workflows/ci.yml).
There are no secret checks.

## Four things that will surprise you

**Ports are `Protocol`, not abstract base classes.** You do not subclass
`LLMProvider`; you write a class with the right shape and it *is* one. That is what
lets a downstream team satisfy a port with a class they already have, and lets tests
substitute a plain object with no mock library.

**The kernel does not know any vendor's name.** There is no
`OpenAIConfig | AnthropicConfig` union anywhere in the core, and adding a provider
must never require editing a core file. Each adapter declares its *own* options model
and validates its own slice. See
[the config module](../modules/configuration.md#adapter-options).

**Vendor SDKs are imported inside factories, never at module level.** `import
anthropic` at the top of an adapter would make `sarb[anthropic]` mandatory for
everyone who installs `sarb`. There is a CI job that installs the core alone and
imports it, purely to catch this.

**`config` is a mapping, not a single value.** `llms.default`, `llms.fast`,
`llms.judge` — production does not run one model. Anything you build should assume
several.

## Where to ask, and when

Ask early on anything that looks like it wants to bend the dependency rule. The rule
is not there to be worked around; if your feature genuinely cannot be expressed within
it, that is a real finding about the ports, and the ports are allowed to change. What
is not allowed is a quiet import that makes the problem disappear.

Everything else: read [conventions](conventions.md), open a draft PR early, and let
the tests tell you what you broke.
