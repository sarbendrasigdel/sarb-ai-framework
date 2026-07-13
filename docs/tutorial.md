# Using sarb

A guided tour for someone who wants to *build something* with this framework, rather
than contribute to it. It assumes you have never seen the codebase and do not intend
to read it.

By the end you will have a running application, and — more usefully — you will know
what happened when you ran it, which is the part that makes the next thing you build
obvious rather than mysterious.

## What this is for

Suppose your company builds four AI products over two years: a support chatbot, a
document-search tool over contracts, an internal assistant, and a coding helper.

Built the usual way, each one grows its own copy of the same plumbing — a client for
whichever model vendor was fashionable that quarter, a chunker, an embedder, a vector
database client, a retry policy, a way to stream tokens to a browser, a way to log what
the model was actually sent. Four copies, each slightly different, each written by
whoever was free.

Then the vendor changes. A better model ships, or the pricing does, or a customer
requires that nothing leaves their network. Now you have four rewrites, in four
codebases, each with its own subtle assumptions about how "the model" behaves baked
into the business logic.

**`sarb` is the layer that stops that happening.** It is a framework, not an
application: it does not answer questions or rank documents. It provides the
infrastructure those four products share, behind interfaces that do not name a vendor —
so your code depends on `LLMProvider`, not on Anthropic; on `VectorStore`, not on
Qdrant.

The concrete promise, which everything in the design is bent towards:

!!! quote "Swapping a vendor is a config change, not a project."
    One line of YAML. No application code is edited, recompiled, or re-tested.

## What you need to know already

Honestly, and in order of how much it matters:

**Required.**

- **Python, comfortably.** Type hints, classes, and `async`/`await` — not deeply, but
  you should not be surprised by `await something()` or by a function you cannot call
  without one. Every port in this framework is async, because streaming tokens and
  calling three tools at once are the normal case.
- **YAML**, to the extent that nested keys hold no fear.
- **The basic vocabulary of LLM applications**: what a *prompt* and a *token* are,
  roughly what an *embedding* is (text turned into a list of numbers, so that similar
  text lands nearby), and what *RAG* means (find relevant text, put it in the prompt).
  You do not need to know how any of it works internally.

**Helpful, not required.**

- A sense of what *dependency injection* means — you ask for a thing by its interface
  and someone else decides what you actually get. If that sentence is unfamiliar, the
  [flow](#what-actually-happened) section below teaches it by example, which is the
  easier way anyway.
- Docker, if you want to run Postgres or Qdrant locally later.

**Explicitly not required.**

- Any particular vendor's SDK. That is the whole point — you will not import one.
- Clean Architecture, SOLID, ports and adapters. Those shaped the framework; you get
  the benefit without having to hold the theory.

## Install

`uv` is the package manager. The core install is deliberately small — it pulls in
Pydantic and PyYAML and nothing else — and each vendor lives behind an *extra*, so a
chatbot that only calls a hosted API never installs a gigabyte of PyTorch:

```bash
uv add sarb                                  # the core
uv add "sarb[anthropic,fastembed]"           # what you actually use
```

You can start with neither, because two providers need no key and no network at all.

## Your first application

Two files.

```yaml title="config/base.yaml"
app:
  name: support-bot

logging:
  level: INFO
  format: console

llms:
  default:
    provider: fake # (1)!
    options:
      responses: ["Paris is the capital of France."]

embedders:
  default:
    provider: fake
    options:
      dimensions: 384
```

1. `fake` is a real, registered provider — not a mock. It answers from a script, with
   no key and no network, and the whole application runs on it. Swapping it for
   `anthropic` later changes this line and nothing else.

```python title="app.py"
import anyio

from sarb.contracts import EMBEDDERS, LLMS, EmbeddingKind, LLMRequest, Message, TextDelta
from sarb.di import bootstrap
from sarb.observability import configure_logging, get_logger

log = get_logger(__name__)


async def main() -> None:
    async with bootstrap(config_dir="config") as container:
        configure_logging(container.settings.logging)

        llm = await container.get(LLMS)
        embedder = await container.get(EMBEDDERS)

        # A complete answer.
        response = await llm.generate(
            LLMRequest(messages=(Message.user("What is the capital of France?"),))
        )
        print("answer:", response.text)
        print("usage :", response.usage.total_tokens, "tokens")

        # Or the same thing, token by token.
        async for event in llm.stream(LLMRequest(messages=(Message.user("hi"),))):
            if isinstance(event, TextDelta):
                print(event.text, end="")

        # Text into vectors, ready for a vector store.
        embedded = await embedder.embed(["Paris is the capital."], kind=EmbeddingKind.DOCUMENT)
        print("vector:", len(embedded.vectors[0]), "dimensions")


anyio.run(main)
```

```console
$ python app.py
INFO  __main__  resolved  embedder='fake' llm='FakeProvider'
answer: Paris is the capital of France.
usage : 14 tokens
stream: Paris is the capital of France.
vector: 384 dimensions
```

Notice what is **not** in `app.py`: the word Anthropic, the word OpenAI, an API key, an
SDK import, a base URL, a retry policy. There is nothing in that file that has to change
when any of those do.

## What actually happened

This is the part worth understanding. Three things collaborated, and each knows exactly
one third of the problem.

```
config/base.yaml     says WHAT to build      llms.default = fake
the registry         says HOW to build it    "fake" → FakeProvider's factory
the DI container     builds it — once — and hands it back by port
```

Step by step, when `bootstrap()` ran and you called `container.get(LLMS)`:

1. **The config was loaded and validated.** `base.yaml`, plus any profile overlay, plus
   environment variables. `${ANTHROPIC_API_KEY}` references are resolved here, and a
   missing one fails the boot immediately — not on the first customer request at 2am.

2. **The container looked up `llms.default`** and found `provider: fake`. It does not
   know what `fake` is. It has never heard of Anthropic either. It just has a string.

3. **The registry turned that string into a factory.** Every adapter registers itself
   under a key — `"anthropic"`, `"openai"`, `"fastembed"`, `"fake"` — and the registry
   is a map from key to a function that can build it.

4. **The factory ran.** *This* is where the vendor's name appears, for the first and only
   time: the factory imports the SDK (and if the extra is not installed, tells you
   exactly which one line to run), validates the `options:` block against a schema the
   *adapter* owns, and constructs the client.

5. **You got back an `LLMProvider`.** Not a `FakeProvider`, as far as your code is
   concerned — an object satisfying an interface with `generate()` and `stream()` on it.
   Ask again and you get the same instance; it holds a connection pool, and rebuilding
   it per request would be a leak with a deadline.

6. **On exit, the container closed it.** That is what the `async with` is for.

The important consequence: **your code and the vendor never meet.** They are introduced
at startup, by the container, on the strength of a string in a config file. Nothing in
`app.py` imported anything vendor-shaped, so nothing in `app.py` has an opinion about
which vendor it got.

### The picture

```
                 your application code
                          │
                          │ asks for a port
                          ▼
   ┌──────────────────────────────────────────────┐
   │  contracts:  LLMProvider, EmbeddingProvider,  │   ← the vocabulary.
   │              VectorStore, Message, Document   │     names no vendor.
   └──────────────────────────────────────────────┘
                          ▲
                          │ implemented by
   ┌──────────────────────────────────────────────┐
   │  providers:  anthropic  openai  gemini        │   ← the adapters.
   │              fastembed  qdrant  fake          │     one vendor each.
   └──────────────────────────────────────────────┘
                          ▲
                          │ chosen by
                 config  +  the DI container
```

Everything above the line depends only on the interface. Everything below it is
replaceable. The line is enforced by an automated test, not by good intentions — see
[the architecture overview](architecture/overview.md) if you want the reasoning.

## The payoff: swapping a vendor

Go back to `config/base.yaml` and change five lines:

```yaml
llms:
  default:
    provider: anthropic # (1)!
    options:
      model: claude-sonnet-5
      api_key: ${ANTHROPIC_API_KEY} # (2)!
```

1. The only thing that changed.
2. A *reference*, not a secret. Config files stay committable; the value comes from the
   environment at boot.

```bash
uv add "sarb[anthropic]"
export ANTHROPIC_API_KEY=sk-...
python app.py            # unchanged. Not one line.
```

`app.py` is not edited. It is not recompiled. Its tests are not rewritten. That is the
whole thesis of the framework, and it is worth trying once, because it is the moment the
architecture stops being an abstraction.

The same move works for `openai`, `gemini`, or `ollama` (a model running on your own
laptop, for when nothing may leave the building).

## Named components

Production does not run one model. It runs a cheap one for routing, an expensive one for
the final answer, and sometimes a third to judge the output of the first two. So every
family is a *map*, not a single value:

```yaml
llms:
  default:
    provider: anthropic
    options: { model: claude-sonnet-5, api_key: ${ANTHROPIC_API_KEY} }
  fast:
    provider: anthropic
    options: { model: claude-haiku-4-5, api_key: ${ANTHROPIC_API_KEY} }
  judge:
    provider: openai
    options: { model: gpt-4o, api_key: ${OPENAI_API_KEY} }
```

```python
llm = await container.get(LLMS)             # "default"
fast = await container.get(LLMS, "fast")
judge = await container.get(LLMS, "judge")  # a different vendor entirely
```

Three models, two vendors, one config file, and application code that still names none
of them.

## Profiles

`base.yaml` holds what is true everywhere. A profile is overlaid on top of it and only
restates what differs:

```yaml title="config/development.yaml"
llms:
  default:
    provider: fake # no key needed, no cost, no network
```

```yaml title="config/production.yaml"
logging:
  format: json # for the log aggregator, not for a human

telemetry:
  enabled: true
```

```python
async with bootstrap(config_dir="config", profile="production") as container:
    ...
```

Which means a developer can run the entire system — every engine, the API, the lot — on
a laptop with no keys and no accounts, and the thing they are running is the same code
that ships.

## What is built, and what is not

`sarb` is being built module by module, and it would be dishonest to show you a RAG
tutorial for engines that do not exist yet. As of today:

| | |
| --- | --- |
| **Ready** | Configuration and profiles, observability (structured logging + OpenTelemetry), the DI container and plugin registry, all the ports, **LLM providers** (Anthropic, OpenAI, Gemini, Ollama, fake), **embedding providers** (OpenAI, FastEmbed, Sentence Transformers, fake) |
| **Next** | Vector stores (Qdrant, pgvector), document loading and chunking, the retrieval engine, memory, tools, the agent engine, the workflow engine, the REST + streaming API, the CLI |

Everything in the first row works today and is what the tutorial above uses. The second
row will arrive behind the same ports, which means the code you write now against
`LLMProvider` and `EmbeddingProvider` will not change when it does.

## Extending it

If a provider you need is not in the box — an internal model gateway, a vector database
of your own — you do not fork the framework and you do not edit a core file. You write a
package that registers itself:

```toml
[project.entry-points."sarb.plugins"]
acme = "acme.sarb:register"
```

Install it, and `provider: acme` works in YAML exactly like the built-in ones do. The
built-in adapters use this same mechanism with no privileged path — if the plugin API
could not express them, that would be a bug in the plugin API. See
[the plugin system](modules/plugins.md).

## Where to go next

- **[Architecture overview](architecture/overview.md)** — the rings, the one rule that
  carries the design, and the five decisions behind it, with the reasoning.
- **[Configuration](modules/configuration.md)** — profiles, secrets, environment
  overrides, and what happens when you typo a key.
- **[LLM providers](modules/llm-providers.md)** and
  **[Embedding providers](modules/embeddings.md)** — every option each adapter takes,
  and the vendor quirks each one absorbs so that you do not have to.
- **[Onboarding](contributing/onboarding.md)** — if you have decided you want to work on
  the framework itself rather than on top of it.
