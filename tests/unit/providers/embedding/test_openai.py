"""Tests for the OpenAI embedding adapter.

Fixtures are built from OpenAI's own ``CreateEmbeddingResponse``, not hand-written
dicts, for the reason the LLM adapter tests give: a fixture you invented can be wrong
in the same way the adapter is wrong, and the two will agree with each other forever.

The client is injected, so none of this touches a network or a key.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openai")

import httpx
import openai
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from sarb.contracts import EmbeddingKind, EmbeddingProvider
from sarb.errors import AuthenticationError, EmbeddingError, ProviderTimeoutError, RateLimitError
from sarb.providers.embedding.openai import OpenAIEmbeddingOptions, OpenAIEmbeddingProvider


class FakeEmbeddings:
    def __init__(self, result):
        self._result = result
        self.calls: list[dict[str, object]] = []

    async def create(self, **params):
        self.calls.append(params)
        if isinstance(self._result, BaseException):
            raise self._result
        if callable(self._result):
            return self._result(params)
        return self._result


class FakeClient:
    def __init__(self, result=None):
        self.embeddings = FakeEmbeddings(result)
        self.closed = False

    async def close(self):
        self.closed = True


def provider(result=None, **options) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        FakeClient(result),  # type: ignore[arg-type]
        OpenAIEmbeddingOptions(model="text-embedding-3-small", dimensions=3, **options),
    )


def sent(llm):
    """Every request the adapter would actually have made."""
    return llm._client.embeddings.calls


def closed(llm):
    """Whether the adapter released its client."""
    return llm._client.closed


def response(*vectors: list[float], shuffled: bool = False) -> CreateEmbeddingResponse:
    """A real OpenAI embedding response, validated by OpenAI's own model."""
    data = [
        Embedding(embedding=vector, index=index, object="embedding")
        for index, vector in enumerate(vectors)
    ]
    if shuffled:
        data.reverse()

    return CreateEmbeddingResponse(
        data=data,
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=4, total_tokens=4),
    )


class TestEmbedding:
    async def test_it_returns_one_vector_per_text(self):
        embedder = provider(response([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))

        vectors = (await embedder.embed(["a", "b"])).vectors

        assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    async def test_order_comes_from_the_index_not_from_the_wire(self):
        # `index` exists on the response precisely because order is a promise the
        # transport makes and not one the protocol does. A permuted batch is the
        # silent failure this whole module is built to prevent, so it is not left to
        # chance: the adapter sorts.
        embedder = provider(response([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], shuffled=True))

        vectors = (await embedder.embed(["first", "second"])).vectors

        assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    async def test_embedding_nothing_makes_no_request(self):
        # A hosted API rejects an empty input, and a document that chunked to zero
        # chunks is an entirely ordinary thing for a caller to end up holding.
        embedder = provider(response())

        assert (await embedder.embed([])).vectors == []
        assert sent(embedder) == []

    async def test_a_large_call_is_split_into_requests_the_api_will_accept(self):
        # The caller does not know, and should not need to know, that OpenAI caps
        # inputs per request. They embed a document; the document is however many
        # chunks it turned out to be.
        embedder = provider(
            lambda params: response(*([[1.0, 0.0, 0.0]] * len(params["input"]))),
            batch_size=2,
        )

        vectors = (await embedder.embed([f"chunk {i}" for i in range(5)])).vectors

        assert len(vectors) == 5
        assert [len(call["input"]) for call in sent(embedder)] == [2, 2, 1]

    async def test_the_kind_is_accepted_and_ignored(self):
        # OpenAI's embedding models are symmetric. The port permits ignoring `kind`,
        # and the caller must not have to know which sort of model they were handed.
        embedder = provider(response([1.0, 0.0, 0.0]))

        await embedder.embed(["a"], kind=EmbeddingKind.QUERY)

        assert "task_type" not in sent(embedder)[0]

    async def test_normalisation_is_applied_when_asked_for(self):
        embedder = provider(response([3.0, 4.0, 0.0]), normalize=True)

        assert (await embedder.embed(["a"])).vectors == [[0.6, 0.8, 0.0]]


class TestRequestTranslation:
    async def test_dimensions_are_sent(self):
        # Which is what makes the configured width binding rather than aspirational:
        # the API truncates to it, and check_output confirms it did.
        embedder = provider(response([1.0, 0.0, 0.0]))

        await embedder.embed(["a"])

        assert sent(embedder)[0]["dimensions"] == 3

    async def test_dimensions_can_be_withheld_for_a_server_that_rejects_them(self):
        # ada-002 and most self-hosted servers on this API shape do. The width check
        # still runs, so a wrong configured value fails on the first call.
        embedder = provider(response([1.0, 0.0, 0.0]), send_dimensions=False)

        await embedder.embed(["a"])

        assert "dimensions" not in sent(embedder)[0]

    async def test_it_asks_for_floats_not_base64(self):
        embedder = provider(response([1.0, 0.0, 0.0]))

        await embedder.embed(["a"])

        assert sent(embedder)[0]["encoding_format"] == "float"


class TestTheOutputIsChecked:
    async def test_a_short_batch_is_caught(self):
        # If this passed, every vector after the missing one would be stored against
        # the wrong text, and nothing in the stack would ever say so.
        embedder = provider(response([1.0, 0.0, 0.0]))

        with pytest.raises(EmbeddingError, match="1 vectors for 2 texts"):
            await embedder.embed(["a", "b"])

    async def test_a_server_that_ignored_the_requested_width_is_caught(self):
        embedder = provider(response([1.0, 0.0, 0.0, 9.0]))

        with pytest.raises(EmbeddingError, match="4-dimensional vector where 3 was configured"):
            await embedder.embed(["a"])


def http_response(status: int, headers: dict[str, str] | None = None):
    """A real ``httpx.Response`` — the vendor's exceptions will not accept less."""
    return httpx.Response(
        status,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
    )


class TestErrors:
    async def test_a_rejected_key_is_not_retryable(self):
        embedder = provider(
            openai.AuthenticationError("bad key", response=http_response(401), body=None)
        )

        with pytest.raises(AuthenticationError) as caught:
            await embedder.embed(["a"])

        assert not caught.value.retryable

    async def test_a_rate_limit_carries_what_the_provider_asked_for(self):
        embedder = provider(
            openai.RateLimitError(
                "slow down",
                response=http_response(429, {"retry-after": "30"}),
                body=None,
            )
        )

        with pytest.raises(RateLimitError) as caught:
            await embedder.embed(["a"])

        assert caught.value.retry_after == 30.0
        assert caught.value.retryable

    async def test_a_server_error_is_an_embedding_error_and_is_retryable(self):
        # The same 500 that would be an LLMError from a chat call. The class differs;
        # the retry semantics do not, which is the point of sharing the status table.
        embedder = provider(
            openai.InternalServerError("boom", response=http_response(503), body=None)
        )

        with pytest.raises(EmbeddingError) as caught:
            await embedder.embed(["a"])

        assert caught.value.retryable

    async def test_a_timeout_is_a_timeout(self):
        embedder = provider(openai.APITimeoutError(request=httpx.Request("POST", "https://x")))

        with pytest.raises(ProviderTimeoutError):
            await embedder.embed(["a"])

    async def test_the_model_is_queryable_on_the_error(self):
        embedder = provider(
            openai.AuthenticationError("bad key", response=http_response(401), body=None)
        )

        with pytest.raises(AuthenticationError) as caught:
            await embedder.embed(["a"])

        assert caught.value.context["model"] == "text-embedding-3-small"


class TestPort:
    def test_it_satisfies_the_port_by_shape(self):
        assert isinstance(provider(), EmbeddingProvider)

    def test_it_reports_what_it_was_configured_with(self):
        embedder = provider()

        assert embedder.model == "text-embedding-3-small"
        assert embedder.dimensions == 3

    async def test_closing_releases_the_connection_pool(self):
        # The container closes every component with an `aclose` on shutdown. A pool
        # that outlives its provider is a leak with a deadline.
        embedder = provider()

        await embedder.aclose()

        assert closed(embedder)
