# sarb

A modular, production-grade framework for building AI agent applications — RAG
systems, chatbots, assistants, document intelligence, workflow agents, coding
agents — without rewriting the infrastructure underneath each one.

`sarb` is a **framework, not an application**. It ships interfaces and adapters:
your product depends on `LLMProvider`, not on OpenAI; on `VectorStore`, not on
Qdrant. Swapping either is a config change, not a rewrite.

## Status

Under active construction. See [`docs/architecture/overview.md`](docs/architecture/overview.md)
for the design, and the module roadmap below for what exists today.

| Module | Status |
| --- | --- |
| Configuration | ✅ |
| Dependency injection + plugins | ✅ |
| Observability (logging, telemetry) | ✅ |
| Contracts (models + ports) | ✅ |
| LLM providers (Anthropic, OpenAI, Gemini, Ollama) | ✅ |
| Embeddings (OpenAI, FastEmbed, Sentence Transformers) | ✅ |
| Middleware (retry, concurrency limits, telemetry) | ✅ |
| Vector stores (Qdrant, pgvector) | ✅ |
| Prompt management | ⬜ |
| Document processing | ⬜ |
| Retrieval engine | ⬜ |
| Memory engine | ⬜ |
| Tool calling | ⬜ |
| Agent engine | ⬜ |
| Workflow engine | ⬜ |
| Storage providers | ⬜ |
| Authentication | ⬜ |
| REST + streaming API | ⬜ |
| Evaluation | ⬜ |
| CLI | ⬜ |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Install

The core is deliberately small. Vendor SDKs and heavy runtimes live behind
extras, so a chatbot that only calls a hosted API never installs a GPU stack:

```bash
uv add sarb                                  # core only
uv add "sarb[anthropic,qdrant,api]"          # what you actually use
```

## Develop

```bash
uv sync --all-extras --group dev   # install everything
uv run pytest                      # tests
uv run ruff check .                # lint
uv run ruff format .               # format
uv run mypy                        # type check (strict)
uv run mkdocs serve                # docs at http://127.0.0.1:8000
```

## The one rule

Dependencies point inward:

```
  api/ cli/       delivery      → engines, contracts
  engines/        use-cases     → contracts ONLY
  contracts/      ports+models  → nothing
  providers/      adapters      → contracts + one vendor SDK
```

**`engines/` must never import `providers/`.** An agent engine knows the
`LLMProvider` port; it does not know that Anthropic exists. Wiring happens once,
from configuration, in the DI container.

This is enforced by [`tests/architecture/`](tests/architecture/), not by good
intentions — an import that breaks the model fails CI.

## Documentation

```bash
uv run mkdocs serve
```

- [Tutorial](docs/tutorial.md) — what it is for, and how to build on it
- [Distribution](docs/distribution.md) — how a project depends on sarb, and how releases are cut
- [Architecture overview](docs/architecture/overview.md)
- [The ports](docs/architecture/ports.md)
- [Configuration](docs/modules/configuration.md)
- [Observability](docs/modules/observability.md)
- [Dependency injection](docs/modules/dependency-injection.md)
- [Plugin system](docs/modules/plugins.md)
- [LLM providers](docs/modules/llm-providers.md)
- [Embedding providers](docs/modules/embeddings.md)
- [Middleware](docs/modules/middleware.md)

## Contributing

New to the codebase, start with [onboarding](docs/contributing/onboarding.md) — the
mental model, what to read in what order, and the local dev loop. Then
[your first contribution](docs/contributing/your-first-contribution.md) walks through
adding a provider end to end. [CONTRIBUTING.md](CONTRIBUTING.md) is the short version.

## Licence

Proprietary.
