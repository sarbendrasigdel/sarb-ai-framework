# sarb

A modular, production-grade framework for building AI agent applications — RAG
systems, chatbots, assistants, document intelligence, workflow agents, coding
agents — without rewriting the infrastructure underneath each one.

`sarb` is a **framework, not an application**. Your product depends on
`LLMProvider`, not on Anthropic; on `VectorStore`, not on Qdrant. Swapping either
is a config change, not a rewrite.

```yaml
llms:
  default:
    provider: anthropic
    options:
      model: claude-sonnet-5
      api_key: ${ANTHROPIC_API_KEY}
```

```python
llm = container.get(LLMProvider)     # your code never names a vendor
```

## Start here

- **[Tutorial](tutorial.md)** — start here if you want to *build* something with
  `sarb`. What it is for, what you need to know first, a running application, and a
  walk through what actually happens when you run it.
- **[Architecture overview](architecture/overview.md)** — the rings, the one rule
  that carries the design, and the five decisions behind it.
- **[Configuration](modules/configuration.md)** — how every module gets its
  settings.
- **[LLM providers](modules/llm-providers.md)** — the first ports with real
  adapters behind them, and the worked example every future adapter follows.
- **[Embedding providers](modules/embeddings.md)** — the same shape against a port
  whose failures are silent, which changes what the adapters have to check.
- **[Middleware](modules/middleware.md)** — retry, concurrency limits and telemetry,
  applied to every component without any adapter or engine knowing they exist.

## Contributing

New to the codebase? **[Onboarding](contributing/onboarding.md)** is written for
someone who has never seen it: the mental model, what to read in what order, and
the local dev loop. Then
**[your first contribution](contributing/your-first-contribution.md)** walks through
adding a provider end to end.

## Install

`sarb` is a package, not a template — you add it as a dependency, you do not clone
it. The core is deliberately small, and vendor SDKs live behind extras, so a chatbot
that only calls a hosted API never installs a GPU stack:

```bash
uv add "sarb @ git+ssh://git@github.com/sarbendra/sarb.git@v0.1.0"
uv add "sarb[anthropic,qdrant,api] @ git+ssh://git@github.com/sarbendra/sarb.git@v0.1.0"
```

See **[Distribution](distribution.md)** for how releases are cut, how a consuming
project is laid out, and how to work on the framework and a product at once.
