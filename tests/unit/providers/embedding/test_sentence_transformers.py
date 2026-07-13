"""Tests for the Sentence Transformers adapter.

The loaded model is injected, so nothing here downloads weights or imports torch.

The assertions that matter are the ones about ``prompt``. Sentence Transformers, unlike
FastEmbed, does not handle the query/document asymmetry for you — the caller of the
port is not allowed to know that, so the adapter has to.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

import numpy as np

from sarb.contracts import EmbeddingKind, EmbeddingProvider
from sarb.errors import EmbeddingError
from sarb.providers.embedding.sentence_transformers import (
    SentenceTransformersOptions,
    SentenceTransformersProvider,
)


class FakeModel:
    """Stands in for a loaded ``SentenceTransformer``."""

    def __init__(self, width: int = 3):
        self._width = width
        self.calls: list[dict[str, object]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._width

    def encode(self, inputs, **params):
        self.calls.append({"inputs": list(inputs), **params})
        return np.array([np.arange(self._width, dtype=np.float32) + i for i in range(len(inputs))])


def provider(model: FakeModel | None = None, **options) -> SentenceTransformersProvider:
    return SentenceTransformersProvider(
        model or FakeModel(),
        SentenceTransformersOptions(model="intfloat/e5-small-v2", **options),
    )


def sent(embedder):
    """Every call the adapter would actually have made into the model."""
    return embedder._model.calls


class TestEmbedding:
    async def test_it_returns_plain_lists_of_floats(self):
        vectors = (await provider().embed(["a"])).vectors

        assert vectors == [[0.0, 1.0, 2.0]]

    async def test_one_vector_per_text(self):
        assert len(await provider().embed(["a", "b"])) == 2

    async def test_embedding_nothing_does_not_wake_the_model(self):
        model = FakeModel()

        assert (await provider(model).embed([])).vectors == []
        assert model.calls == []

    async def test_the_width_is_read_from_the_loaded_model(self):
        # No table of model names to widths, here or anywhere. The model knows.
        assert provider(FakeModel(width=768)).dimensions == 768

    async def test_normalisation_is_delegated_to_the_library(self):
        # Same arithmetic, done in numpy over a batch instead of in our own Python
        # over a loop.
        embedder = provider(normalize=True)

        await embedder.embed(["a"])

        assert sent(embedder)[0]["normalize_embeddings"] is True


class TestAsymmetry:
    async def test_a_query_gets_the_query_prompt(self):
        # E5 wants "query: " on one side and "passage: " on the other. Get it the
        # wrong way round and retrieval still works, just worse — which is the whole
        # reason EmbeddingKind is on the port instead of being left to the caller.
        embedder = provider(query_prompt="query: ", document_prompt="passage: ")

        await embedder.embed(["what is the capital?"], kind=EmbeddingKind.QUERY)

        assert sent(embedder)[0]["prompt"] == "query: "

    async def test_a_document_gets_the_document_prompt(self):
        embedder = provider(query_prompt="query: ", document_prompt="passage: ")

        await embedder.embed(["Paris is the capital."], kind=EmbeddingKind.DOCUMENT)

        assert sent(embedder)[0]["prompt"] == "passage: "

    async def test_a_symmetric_model_is_given_no_prompt_at_all(self):
        # Both prompts default to empty, and an empty prefix must become `None` rather
        # than `""` — prefixing a model that was not trained with a prefix makes
        # retrieval worse, and "" is a prefix.
        embedder = provider()

        await embedder.embed(["a"])

        assert sent(embedder)[0]["prompt"] is None


class TestTheOutputIsChecked:
    async def test_a_model_that_returned_the_wrong_width_is_caught(self):
        class Wrong(FakeModel):
            def encode(self, inputs, **params):
                return np.array([np.zeros(9, dtype=np.float32) for _ in inputs])

        with pytest.raises(EmbeddingError, match="9-dimensional vector where 3 was configured"):
            await provider(Wrong()).embed(["a"])


class TestErrors:
    async def test_a_torch_failure_becomes_an_embedding_error(self):
        class Broken(FakeModel):
            def encode(self, inputs, **params):
                raise RuntimeError("CUDA out of memory")

        with pytest.raises(EmbeddingError, match="CUDA out of memory") as caught:
            await provider(Broken()).embed(["a"])

        assert not caught.value.retryable


class TestPort:
    def test_it_satisfies_the_port_by_shape(self):
        assert isinstance(provider(), EmbeddingProvider)

    async def test_the_configured_batch_size_reaches_the_model(self):
        embedder = provider(batch_size=8)

        await embedder.embed(["a"])

        assert sent(embedder)[0]["batch_size"] == 8
