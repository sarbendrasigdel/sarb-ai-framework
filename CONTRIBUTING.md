# Contributing

## Setup

```bash
uv sync --all-extras --group dev
uv run pytest
```

If that is green, you are set up.

## The loop

```bash
uv run pytest                                # ~12 seconds. Run it constantly
uv run ruff check --fix . && uv run ruff format .
uv run mypy                                  # strict
uv run --group docs mkdocs serve             # docs are part of the change
```

Before you open a PR, run all four plus `uv run --group docs mkdocs build --strict`.
That is exactly what CI runs — there are no hidden checks.

## Read this first

Full documentation is in `docs/`, and `uv run --group docs mkdocs serve` is the nicest
way to read it. Start with:

- **[Onboarding](docs/contributing/onboarding.md)** — the mental model, what to read in
  what order, and the four things about this codebase that will surprise you. Twenty
  minutes, and it will save you more.
- **[Architecture overview](docs/architecture/overview.md)** — the rings, and the five
  decisions behind them.
- **[Your first contribution](docs/contributing/your-first-contribution.md)** — adding a
  provider, end to end.
- **[Testing](docs/contributing/testing.md)** and
  **[Conventions](docs/contributing/conventions.md)** — including the architecture tests
  that will fail your PR, and why they exist.

## The one rule

**`engines/` may never import `providers/`.**

Business logic depends on a port — `LLMProvider`, `VectorStore` — never on a vendor.
That is what makes swapping Anthropic for OpenAI, or Qdrant for pgvector, a config
change rather than a rewrite, and it is the reason this framework exists at all.

It is enforced by `tests/architecture/`, which parses every module and fails CI on
violation. If a change you want to make seems to require breaking it, say so — that is
a real finding about the ports, and the ports are allowed to change. A quiet import
that makes the problem disappear is not.
