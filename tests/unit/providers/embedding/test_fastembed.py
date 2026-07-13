"""Tests for the FastEmbed adapter.

The model is injected, so these run in milliseconds and download nothing. That is not
only convenience: a test suite that pulls a 130MB ONNX file is a test suite people
stop running.

What *is* tested against the real library is :func:`dimensions_for` — the one place
the adapter depends on FastEmbed's own knowledge of its own models, and therefore the
one place a hand-written stub would be able to lie to us.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastembed")

import fastembed
import numpy as np

from sarb.contracts import EmbeddingKind, EmbeddingProvider
from sarb.errors import EmbeddingError
from sarb.providers.embedding.fastembed import (
    FastEmbedOptions,
    FastEmbedProvider,
    dimensions_for,
)


class FakeModel:
    """Stands in for ``fastembed.TextEmbedding``, recording which path was taken."""

    def __init__(self, width: int = 3):
        self._width = width
        self.embedded: list[tuple[str, list[str]]] = []
        self.batch_sizes: list[int] = []

    def embed(self, texts, batch_size=256, **_):
        self.embedded.append(("document", list(texts)))
        self.batch_sizes.append(batch_size)
        return (np.arange(self._width, dtype=np.float32) + i for i in range(len(texts)))

    def query_embed(self, texts, batch_size=256, **_):
        self.embedded.append(("query", list(texts)))
        self.batch_sizes.append(batch_size)
        return (np.arange(self._width, dtype=np.float32) + i for i in range(len(texts)))


def provider(model: FakeModel | None = None, *, dimensions: int = 3, **options):
    model = model or FakeModel()
    return FastEmbedProvider(
        model,
        FastEmbedOptions(model="BAAI/bge-small-en-v1.5", **options),
        dimensions=dimensions,
    )


class TestEmbedding:
    async def test_it_returns_a_plain_list_of_floats(self):
        # Not a numpy array. `contracts` depends on Pydantic and the standard library
        # and nothing else, so the array stops here.
        vectors = (await provider().embed(["a"])).vectors

        assert vectors == [[0.0, 1.0, 2.0]]
        assert all(isinstance(value, float) for value in vectors[0])

    async def test_one_vector_per_text(self):
        assert len(await provider().embed(["a", "b", "c"])) == 3

    async def test_embedding_nothing_does_not_wake_the_model(self):
        model = FakeModel()

        assert (await provider(model).embed([])).vectors == []
        assert model.embedded == []

    async def test_normalisation_is_applied_when_asked_for(self):
        model = FakeModel(width=2)
        embedder = provider(model, dimensions=2, normalize=True)

        # The model's first vector is [0, 1]; its second is [1, 2].
        vectors = (await embedder.embed(["a", "b"])).vectors

        assert vectors[0] == [0.0, 1.0]
        assert vectors[1] == pytest.approx([0.4472135955, 0.894427191])


class TestAsymmetry:
    async def test_a_query_takes_the_query_path(self):
        # The BGE family is asymmetric: it was trained with one prefix on the thing
        # being searched for and another on the thing being searched. Send a query
        # down the document path and nothing errors — recall just quietly drops, for
        # reasons no traceback will ever mention.
        model = FakeModel()

        await provider(model).embed(["what is the capital?"], kind=EmbeddingKind.QUERY)

        assert model.embedded[0][0] == "query"

    async def test_a_document_takes_the_document_path(self):
        model = FakeModel()

        await provider(model).embed(["Paris is the capital."], kind=EmbeddingKind.DOCUMENT)

        assert model.embedded[0][0] == "document"

    async def test_documents_are_the_default(self):
        # Indexing is the bulk of what anyone embeds, and the port agrees.
        model = FakeModel()

        await provider(model).embed(["Paris is the capital."])

        assert model.embedded[0][0] == "document"


class TestTheOutputIsChecked:
    async def test_a_model_of_the_wrong_width_is_caught(self):
        # The custom-model case: someone set `dimensions` by hand and got it wrong.
        # Better to fail on the first call than to create a collection around a lie.
        embedder = provider(FakeModel(width=5), dimensions=3)

        with pytest.raises(EmbeddingError, match="5-dimensional vector where 3 was configured"):
            await embedder.embed(["a"])


class TestErrors:
    async def test_a_model_failure_becomes_an_embedding_error(self):
        class Broken(FakeModel):
            def embed(self, texts, batch_size=256, **_):
                raise RuntimeError("onnxruntime fell over")

        with pytest.raises(EmbeddingError, match="onnxruntime fell over") as caught:
            await provider(Broken()).embed(["a"])

        # In-process failures do not get better by being retried. An ONNX error, an
        # out-of-memory and bad input are all permanent until something changes.
        assert not caught.value.retryable


class TestDimensionsComeFromTheLibrary:
    def test_fastembed_is_asked_rather_than_a_table_of_our_own(self):
        # The distinction this framework holds to: a vendor may know about its own
        # models; we may not, because our copy of that knowledge goes stale in silence.
        assert dimensions_for(fastembed, "BAAI/bge-small-en-v1.5") == 384

    def test_an_unknown_model_returns_nothing_rather_than_a_guess(self):
        assert dimensions_for(fastembed, "acme/not-a-real-model") is None


class TestPort:
    def test_it_satisfies_the_port_by_shape(self):
        assert isinstance(provider(), EmbeddingProvider)

    def test_it_reports_the_width_it_was_built_with(self):
        assert provider(dimensions=384).dimensions == 384
