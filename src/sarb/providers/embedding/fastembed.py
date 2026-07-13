"""FastEmbed — local ONNX embeddings, on the CPU, with no torch.

The reason this adapter exists at all: it is the only way to run good retrieval with
no API key, no GPU and no gigabyte of PyTorch. A 130MB ONNX model on a CPU is enough
for a great many RAG systems, and being able to say so — in a config file, without
changing a line of application code — is most of what this framework is for.

It is also the first adapter that wraps a **blocking, in-process model** rather than
an HTTP API, which changes two things: the model is loaded at build time, and every
call is run in a worker thread.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import Field

from sarb.config import parse_options
from sarb.contracts import EmbeddingKind, EmbeddingResponse, Vector
from sarb.dependencies import import_optional
from sarb.errors import ConfigurationError, EmbeddingError, SarbError
from sarb.providers.embedding._shared import (
    LocalEmbeddingOptions,
    ThreadRunner,
    check_output,
    normalized,
)

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from sarb.plugins.registry import ComponentContext

__all__ = ["FastEmbedOptions", "FastEmbedProvider", "build"]

PROVIDER = "fastembed"
EXTRA = "fastembed"


class FastEmbedOptions(LocalEmbeddingOptions):
    """Options for the FastEmbed adapter.

    Attributes:
        model: A model FastEmbed knows, e.g. ``BAAI/bge-small-en-v1.5``.
        cuda: Use the GPU. Needs ``onnxruntime-gpu``, which is not in the extra.
        threads: ONNX intra-op threads. Leave unset unless you have measured.
        dimensions: Only for a model FastEmbed cannot tell us the width of.

    Why ``dimensions`` is optional here and required for the hosted adapter:
        FastEmbed ships the widths of the models it supports —
        ``TextEmbedding.list_supported_models()`` returns them — so the adapter reads
        the width from the library rather than from a table of its own. That is the
        distinction the framework holds to: a vendor is allowed to know about its own
        models; *we* are not, because our copy of that knowledge goes stale silently.

        The override is for a custom model FastEmbed has been pointed at but does not
        have metadata for. Setting it when the library does know is harmless: the
        width check compares the two, so a wrong override fails loudly on the first
        call rather than corrupting an index quietly.
    """

    cuda: bool = False
    threads: int | None = Field(default=None, gt=0)
    dimensions: int | None = Field(default=None, gt=0)


class FastEmbedProvider:
    """The :class:`~sarb.contracts.EmbeddingProvider` port, backed by FastEmbed.

    The model is injected, already loaded. :func:`build` does the loading, which is
    slow and downloads weights — and doing it there means a mistyped model name fails
    at startup, in front of whoever deployed it, rather than in front of the first
    user.
    """

    def __init__(self, model: Any, options: FastEmbedOptions, *, dimensions: int) -> None:
        self._model = model
        self._options = options
        self._dimensions = dimensions
        self._runner = ThreadRunner(options.max_concurrency)

    @property
    def model(self) -> str:
        """The embedding model in use."""
        return self._options.model

    @property
    def dimensions(self) -> int:
        """The vector width, as reported by FastEmbed for this model."""
        return self._dimensions

    async def embed(
        self,
        texts: Sequence[str],
        *,
        kind: EmbeddingKind = EmbeddingKind.DOCUMENT,
    ) -> EmbeddingResponse:
        """Embed a batch of texts, off the event loop.

        ``kind`` selects between FastEmbed's two entry points. Several of the models
        it ships — the BGE family especially — are **asymmetric**: they were trained
        with one prefix on the thing being searched for and a different one on the
        thing being searched. Send a query down the document path and nothing errors;
        the vector simply lands slightly off, and recall drops for reasons no
        traceback will ever mention.

        The response reports no tokens. That is not an omission: this model runs on your
        own CPU, and no number of tokens through it produces an invoice. A local
        embedder costs wall-clock and memory, both of which are already on the span.
        """
        if not texts:
            return EmbeddingResponse(vectors=[], model=self._options.model)

        embed = self._model.query_embed if kind is EmbeddingKind.QUERY else self._model.embed
        batch = list(texts)

        def work() -> list[Vector]:
            # `embed` is lazy: it returns a generator, and the ONNX session only runs
            # as it is consumed. The `list()` is not a formality — without it the work
            # would escape the worker thread and happen back on the event loop.
            return [vector.tolist() for vector in embed(batch, batch_size=self._options.batch_size)]

        try:
            vectors = await self._runner.run(work)
        except Exception as exc:
            raise self._fail(exc) from exc

        if self._options.normalize:
            vectors = [normalized(vector) for vector in vectors]

        return EmbeddingResponse(
            vectors=check_output(
                vectors,
                texts,
                provider=PROVIDER,
                model=self._options.model,
                dimensions=self._dimensions,
            ),
            model=self._options.model,
        )

    def _fail(self, exc: Exception) -> SarbError:
        """Translate a FastEmbed failure into one of ours.

        There is no status code to switch on: the model is in this process, so a
        failure is an ONNX error, an out-of-memory, or bad input. None of those get
        better by being retried, so none of them are marked retryable.
        """
        if isinstance(exc, SarbError):
            return exc

        return EmbeddingError(
            f"FastEmbed failed to embed: {exc}",
            provider=PROVIDER,
            context={"model": self._options.model},
            cause=exc,
        )


def dimensions_for(sdk: Any, model: str) -> int | None:
    """Ask FastEmbed how wide this model's vectors are.

    Returns ``None`` for a model it has no metadata for, which is the caller's cue to
    fall back to the configured override.
    """
    for supported in sdk.TextEmbedding.list_supported_models():
        if supported.get("model") == model:
            dim = supported.get("dim")
            return int(dim) if dim else None
    return None


def build(ctx: ComponentContext) -> FastEmbedProvider:
    """Build the ``fastembed`` embedder, loading the model now rather than later."""
    sdk = import_optional("fastembed", extra=EXTRA, used_by=f"embedder {PROVIDER!r}")
    options = parse_options(ctx.config, FastEmbedOptions, component=ctx.path)

    dimensions = options.dimensions or dimensions_for(sdk, options.model)
    if dimensions is None:
        # Refusing to start is the right answer. The width is what the vector-store
        # collection gets created with, so a provider that cannot state it up front
        # can only discover it by embedding something — and by then the collection
        # exists, at whatever width somebody guessed.
        available = ", ".join(
            sorted(m["model"] for m in sdk.TextEmbedding.list_supported_models())[:5]
        )
        raise ConfigurationError(
            f"FastEmbed does not know the vector width of model {options.model!r}. "
            f"Set {ctx.path}.options.dimensions explicitly, or use a supported model "
            f"(e.g. {available}, ...).",
            context={"component": ctx.path, "model": options.model},
        )

    model = sdk.TextEmbedding(
        model_name=options.model,
        cache_dir=options.cache_dir,
        threads=options.threads,
        cuda=options.cuda,
    )
    return FastEmbedProvider(model, options, dimensions=dimensions)
