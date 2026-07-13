"""OpenAI embeddings — and every API-compatible server.

``/v1/embeddings`` is the de facto standard shape, so this one adapter also serves
Hugging Face's text-embeddings-inference, vLLM, LiteLLM, Together, Jina and anything
else that copied it. Point ``base_url`` at them and the rest of this file applies
unchanged.

The interesting decision here is that ``dimensions`` is **required**, which is
unusual enough to be worth reading the class docstring for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import Field, SecretStr

from sarb.config import parse_options
from sarb.contracts import EmbeddingKind, EmbeddingResponse, Usage, Vector
from sarb.dependencies import import_optional
from sarb.errors import EmbeddingError, ProviderTimeoutError, SarbError
from sarb.providers._shared import error_for_status, retry_after_seconds
from sarb.providers.embedding._shared import (
    EmbeddingOptions,
    batches,
    check_output,
    normalized,
)

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from openai import AsyncOpenAI

    from sarb.plugins.registry import ComponentContext

__all__ = ["OpenAIEmbeddingOptions", "OpenAIEmbeddingProvider", "build"]

PROVIDER = "openai"
EXTRA = "openai"


class OpenAIEmbeddingOptions(EmbeddingOptions):
    """Options for the OpenAI embedding adapter.

    Attributes:
        api_key: Read from ``OPENAI_API_KEY`` when absent.
        base_url: Point this at any API-compatible embedding server.
        organization: OpenAI's billing scope.
        timeout: Per-request deadline, in seconds.
        max_retries: Transport retries performed by the SDK itself.
        dimensions: The vector width. **Required** — see below.
        send_dimensions: Whether to pass ``dimensions`` to the API.

    Why ``dimensions`` is required, when the model already has a native width:
        Because with ``text-embedding-3-*`` the width is a *request parameter*, not a
        fact about the model — the vectors are Matryoshka embeddings and can be
        truncated to whatever you ask for. So there is no single right answer to
        default to, and the framework guessing one would be guessing at the shape of
        your index.

        Stating it also buys a real check. The width is what the vector-store
        collection was created with, and
        :func:`~sarb.providers.embedding._shared.check_output` compares every returned
        vector against it — which catches both a model change that was not carried
        through to the index, and a server that ignored the ``dimensions`` it was sent.

    Why ``send_dimensions`` exists:
        Not every server on this API shape supports truncation. ``text-embedding-ada-002``
        rejects the parameter outright, and most self-hosted servers ignore or reject
        it too. Turning it off sends the request without it and lets the width check
        confirm you configured the model's native size. The alternative — a table of
        which models support the parameter — is a table that is wrong the week after
        it is written, and this framework does not keep those.
    """

    api_key: SecretStr | None = None
    base_url: str | None = None
    organization: str | None = None
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    dimensions: int = Field(gt=0)
    send_dimensions: bool = True

    # Hosted APIs accept large batches, and the round trip dominates. 2048 is
    # OpenAI's documented cap on inputs per request; this stays clear of it.
    batch_size: int = Field(default=256, gt=0)


class OpenAIEmbeddingProvider:
    """The :class:`~sarb.contracts.EmbeddingProvider` port, backed by OpenAI.

    The client is injected rather than constructed, so this class can be tested
    against OpenAI's own response models without a key or a network. :func:`build`
    is where the client is made.
    """

    def __init__(self, client: AsyncOpenAI, options: OpenAIEmbeddingOptions) -> None:
        self._client = client
        self._options = options
        self._sdk = import_optional("openai", extra=EXTRA, used_by=f"embedder {PROVIDER!r}")

    @property
    def model(self) -> str:
        """The embedding model in use."""
        return self._options.model

    @property
    def dimensions(self) -> int:
        """The vector width, as configured."""
        return self._options.dimensions

    async def embed(
        self,
        texts: Sequence[str],
        *,
        kind: EmbeddingKind = EmbeddingKind.DOCUMENT,
    ) -> EmbeddingResponse:
        """Embed a batch of texts.

        ``kind`` is accepted and ignored: OpenAI's embedding models are **symmetric**,
        with no query/document distinction to make. The port permits this explicitly,
        and it is the reason the parameter exists on the port rather than being left
        to the models that need it — the caller must not have to know which kind of
        model it was handed.
        """
        del kind  # symmetric model; see docstring

        model = self._options.model

        if not texts:
            # No call, no cost. A hosted API would reject an empty input anyway, and
            # embedding nothing is a perfectly ordinary thing for a caller to end up
            # doing when a document chunks to zero chunks.
            return EmbeddingResponse(vectors=[], model=model)

        vectors: list[Vector] = []
        tokens = 0

        # One `texts` can become several requests, so the tokens are summed rather than
        # taken from the last response — which would report a whole corpus as the cost
        # of its final batch.
        for batch in batches(texts, self._options.batch_size):
            batch_vectors, batch_tokens = await self._embed_batch(batch)
            vectors.extend(batch_vectors)
            tokens += batch_tokens

        if self._options.normalize:
            vectors = [normalized(vector) for vector in vectors]

        return EmbeddingResponse(
            vectors=check_output(
                vectors,
                texts,
                provider=PROVIDER,
                model=model,
                dimensions=self._options.dimensions,
            ),
            model=model,
            usage=Usage(input_tokens=tokens),
        )

    async def _embed_batch(self, texts: Sequence[str]) -> tuple[list[Vector], int]:
        """One request. Returns its vectors in the order the texts were given, and its tokens."""
        params: dict[str, Any] = {
            "model": self._options.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        if self._options.send_dimensions:
            params["dimensions"] = self._options.dimensions

        try:
            response = await self._client.embeddings.create(**params)
        except Exception as exc:
            raise self._fail(exc) from exc

        # Sorted by the API's own `index`, not trusted to arrive in order. The field
        # exists precisely because order is a promise the transport makes and not one
        # the protocol does, and a permuted batch is the silent failure this whole
        # module is built to prevent.
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        tokens = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0

        return vectors, tokens

    def _fail(self, exc: Exception) -> SarbError:
        """Translate an OpenAI exception into one of ours."""
        if isinstance(exc, SarbError):
            return exc

        model = self._options.model

        if isinstance(exc, self._sdk.APITimeoutError):
            return ProviderTimeoutError(
                f"OpenAI embeddings timed out after {self._options.timeout}s",
                provider=PROVIDER,
                context={"model": model},
                cause=exc,
            )

        if isinstance(exc, self._sdk.APIStatusError):
            return error_for_status(
                exc.status_code,
                str(exc),
                provider=PROVIDER,
                model=model,
                otherwise=EmbeddingError,
                retry_after=retry_after_seconds(getattr(exc.response, "headers", None)),
                cause=exc,
            )

        if isinstance(exc, self._sdk.APIConnectionError):
            return EmbeddingError(
                f"Could not reach the OpenAI embeddings API: {exc}",
                provider=PROVIDER,
                context={"model": model},
                cause=exc,
                retryable=True,
            )

        return EmbeddingError(
            f"OpenAI embedding call failed: {exc}",
            provider=PROVIDER,
            context={"model": model},
            cause=exc,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.close()


def build(ctx: ComponentContext) -> OpenAIEmbeddingProvider:
    """Build the ``openai`` embedder from configuration."""
    sdk = import_optional("openai", extra=EXTRA, used_by=f"embedder {PROVIDER!r}")
    options = parse_options(ctx.config, OpenAIEmbeddingOptions, component=ctx.path)

    client = sdk.AsyncOpenAI(
        api_key=options.api_key.get_secret_value() if options.api_key else None,
        organization=options.organization,
        base_url=options.base_url,
        timeout=options.timeout,
        max_retries=options.max_retries,
    )
    return OpenAIEmbeddingProvider(client, options)
