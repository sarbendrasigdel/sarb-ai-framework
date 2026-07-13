"""Tests for the fake embedder.

Everything downstream — the retrieval engine, the memory engine, the indexing
pipeline — will be tested against this provider. A fake that quietly does the wrong
thing produces green tests for broken code, which is worse than having no tests, so it
gets tested as carefully as the real ones.
"""

from __future__ import annotations

import math

import pytest

from sarb.contracts import EmbeddingKind, EmbeddingProvider
from sarb.errors import AuthenticationError, EmbeddingError, ProviderTimeoutError, RateLimitError
from sarb.providers.embedding.fake import FakeEmbeddingOptions, FakeEmbeddingProvider


def provider(**options) -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(FakeEmbeddingOptions(**options))


class TestDeterminism:
    async def test_the_same_text_always_gets_the_same_vector(self):
        assert (await provider().embed(["Paris"])).vectors == (
            await provider().embed(["Paris"])
        ).vectors

    async def test_different_texts_get_different_vectors(self):
        [paris] = (await provider().embed(["Paris"])).vectors
        [rome] = (await provider().embed(["Rome"])).vectors

        assert paris != rome

    async def test_the_vector_does_not_depend_on_which_process_asked(self):
        # Seeded from SHA-256, not from `hash()`, whose salt changes per process. A
        # "deterministic" fake that answers differently on the next run is worse than
        # an obviously random one: it makes a flaky test look like a real failure.
        #
        # This is the value for "Paris" at 4 dimensions. If it ever changes, the fake
        # has stopped being reproducible and every downstream retrieval test that
        # asserted an ordering has quietly become a lottery.
        [vector] = (await provider(dimensions=4).embed(["Paris"])).vectors

        assert vector == pytest.approx(
            [0.4443840523, 0.1282774693, -0.8571192116, 0.2267473528], abs=1e-9
        )

    async def test_vectors_are_unit_length(self):
        # So that cosine and dot product agree, and a retrieval test asserting an
        # ordering means the same thing whichever metric the store was configured for.
        [vector] = (await provider(dimensions=16).embed(["Paris"])).vectors

        assert math.isclose(sum(v * v for v in vector), 1.0)


class TestScriptedGeometry:
    async def test_a_scripted_vector_is_used_verbatim(self):
        # Hashed vectors are deterministic but arbitrary — you cannot ask for "a
        # document near this query" without knowing where the hash happened to put it.
        # Scripting lets a retrieval test say what it means.
        embedder = provider(dimensions=2, vectors={"north": (0.0, 1.0)})

        assert (await embedder.embed(["north"])).vectors == [[0.0, 1.0]]

    async def test_a_scripted_vector_is_normalised(self):
        embedder = provider(dimensions=2, vectors={"north": (0.0, 5.0)})

        assert (await embedder.embed(["north"])).vectors == [[0.0, 1.0]]

    async def test_unscripted_texts_still_fall_back_to_the_hash(self):
        embedder = provider(dimensions=2, vectors={"north": (0.0, 1.0)})

        vectors = (await embedder.embed(["north", "elsewhere"])).vectors

        assert vectors[0] == [0.0, 1.0]
        assert len(vectors[1]) == 2


class TestItRecordsWhatItWasAsked:
    async def test_it_records_the_texts(self):
        embedder = provider()

        await embedder.embed(["a", "b"])

        assert embedder.texts == ["a", "b"]

    async def test_it_records_how_each_text_was_embedded(self):
        # The assertion that catches the nastiest bug in this module. An indexing
        # pipeline that embeds a query as a document produces a system that *works* —
        # just worse — and no test that only inspects the vectors will ever notice.
        embedder = provider()

        await embedder.embed(["a document"])
        await embedder.embed(["a query"], kind=EmbeddingKind.QUERY)

        assert embedder.kinds == [EmbeddingKind.DOCUMENT, EmbeddingKind.QUERY]


class TestFailure:
    @pytest.mark.parametrize(
        ("failure", "expected"),
        [
            ("rate_limit", RateLimitError),
            ("auth", AuthenticationError),
            ("timeout", ProviderTimeoutError),
            ("error", EmbeddingError),
        ],
    )
    async def test_it_can_be_told_to_fail(self, failure, expected):
        # The only honest way to test a retry policy around embedding. The alternative
        # is getting rate limited by a real vendor, deliberately, in CI.
        embedder = provider(fail_with=failure)

        with pytest.raises(expected):
            await embedder.embed(["a"])

    async def test_a_failed_call_is_still_recorded(self):
        embedder = provider(fail_with="error")

        with pytest.raises(EmbeddingError):
            await embedder.embed(["a"])

        assert embedder.texts == ["a"]


class TestPort:
    def test_it_satisfies_the_port_by_shape(self):
        assert isinstance(provider(), EmbeddingProvider)

    async def test_embedding_nothing_returns_nothing(self):
        assert (await provider().embed([])).vectors == []

    async def test_every_vector_is_the_configured_width(self):
        vectors = (await provider(dimensions=384).embed(["a", "b"])).vectors

        assert [len(v) for v in vectors] == [384, 384]
