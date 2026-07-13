"""Registration of the adapters that ship with the framework.

First-party adapters use exactly the same mechanism third parties do — a
``register(registry)`` function that adds factories. The only difference is that
this one is called directly rather than discovered through an entry point, which
saves a metadata scan on every boot and keeps the framework working in packaging
setups where entry points are unreliable (some editable installs, frozen apps).

No privileged path: if the plugin API were not expressive enough to register the
built-ins, that would be a bug in the plugin API, and this module is where it
would show up first.

The rule for everything added here
----------------------------------
**Never import a vendor SDK at module level.** Registration records a factory; it
does not run one. So a registration line looks like::

    registry.add("llms", "anthropic", anthropic.build, summary="Claude models")

where ``anthropic.build`` is a plain module-level function whose *body* does the
``import anthropic``. Importing this module must therefore stay cheap and safe on
a core-only install: every adapter is registered whether or not its extra is
present, and asking for one you have not installed fails at construction with
``uv add "sarb[anthropic]"`` rather than an ``ImportError`` from deep in the
import graph.

That is also why every adapter is registered unconditionally rather than guarded
by a "is the SDK importable?" check. A conditional registration would make
``sarb providers list`` show a different set of providers on every machine, and
turn a missing extra — which has a one-line fix — into "no provider 'anthropic'
registered", which sounds like a bug in the framework.
"""

from __future__ import annotations

from sarb.plugins.registry import ComponentRegistry
from sarb.providers.embedding import fake as fake_embedder
from sarb.providers.embedding import fastembed, sentence_transformers
from sarb.providers.embedding import openai as openai_embedder
from sarb.providers.llm import anthropic, fake, gemini, ollama, openai
from sarb.providers.vectorstore import memory as memory_store
from sarb.providers.vectorstore import pgvector, qdrant

__all__ = ["register"]


def register(registry: ComponentRegistry) -> None:
    """Register every adapter that ships with the framework.

    Args:
        registry: The registry to populate.
    """
    registry.add(
        "llms",
        "anthropic",
        anthropic.build,
        summary="Claude, via the Anthropic Messages API. Needs sarb[anthropic].",
    )
    registry.add(
        "llms",
        "openai",
        openai.build,
        summary="OpenAI Chat Completions, and any API-compatible server. Needs sarb[openai].",
    )
    registry.add(
        "llms",
        "gemini",
        gemini.build,
        summary="Gemini, via Google's Gen AI SDK. Needs sarb[gemini].",
    )
    registry.add(
        "llms",
        "ollama",
        ollama.build,
        summary="A local Ollama daemon, over its native HTTP API. Needs sarb[ollama].",
    )
    registry.add(
        "llms",
        "fake",
        fake.build,
        summary="A scripted model. No network and no key — for tests and local work.",
    )

    registry.add(
        "embedders",
        "openai",
        openai_embedder.build,
        summary="OpenAI embeddings, and any API-compatible server. Needs sarb[openai].",
    )
    registry.add(
        "embedders",
        "fastembed",
        fastembed.build,
        summary="Local ONNX embeddings on the CPU. No key, no torch. Needs sarb[fastembed].",
    )
    registry.add(
        "embedders",
        "sentence-transformers",
        sentence_transformers.build,
        summary="Local PyTorch embeddings. Needs sarb[sentence-transformers].",
    )
    registry.add(
        "embedders",
        "fake",
        fake_embedder.build,
        summary="Deterministic vectors from a hash. No model — for tests and local work.",
    )

    registry.add(
        "vector_stores",
        "qdrant",
        qdrant.build,
        summary="Qdrant, purpose-built for this. Needs sarb[qdrant].",
    )
    registry.add(
        "vector_stores",
        "pgvector",
        pgvector.build,
        summary="Vectors in the Postgres you already run. Needs sarb[pgvector].",
    )
    registry.add(
        "vector_stores",
        "memory",
        memory_store.build,
        summary="Exhaustive search, in this process. No server — for tests and local work.",
    )
