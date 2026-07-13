"""Embedding adapters.

| ``provider:`` | Extra | What it is |
| --- | --- | --- |
| ``openai`` | ``sarb[openai]`` | OpenAI embeddings, and any API-compatible server |
| ``fastembed`` | ``sarb[fastembed]`` | Local ONNX. CPU, no torch, no key |
| ``sentence-transformers`` | ``sarb[sentence-transformers]`` | Local PyTorch, GPU, fine-tunes |
| ``fake`` | — | Deterministic vectors from a hash. No model at all |

Two conventions, the same ones the LLM adapters follow:

* the vendor SDK is imported **lazily, inside the factory**, so the core stays
  installable without it;
* the thing that does the work — an HTTP client, or a loaded model — is **injected**,
  not constructed by the adapter, so translation can be tested with no network and no
  weights on disk.

And one that is specific to this family: a local model is loaded in :func:`build`, at
startup, on purpose. It is slow and it touches the network, and both of those are
better in front of the person deploying than in front of the first user.
"""

from __future__ import annotations

from sarb.providers.embedding.fake import FakeEmbeddingOptions, FakeEmbeddingProvider
from sarb.providers.embedding.fastembed import FastEmbedOptions, FastEmbedProvider
from sarb.providers.embedding.openai import OpenAIEmbeddingOptions, OpenAIEmbeddingProvider
from sarb.providers.embedding.sentence_transformers import (
    SentenceTransformersOptions,
    SentenceTransformersProvider,
)

__all__ = [
    "FakeEmbeddingOptions",
    "FakeEmbeddingProvider",
    "FastEmbedOptions",
    "FastEmbedProvider",
    "OpenAIEmbeddingOptions",
    "OpenAIEmbeddingProvider",
    "SentenceTransformersOptions",
    "SentenceTransformersProvider",
]
