"""Sentence Transformers — local embeddings on PyTorch.

The heavier of the two local adapters, and the one people reach for when they have a
GPU, a fine-tuned model, or a model that has not been exported to ONNX. It is the
research default, so it is where the interesting checkpoints appear first.

Its awkward corner is the query/document asymmetry, which — unlike FastEmbed — it
does not handle for you. See :attr:`SentenceTransformersOptions.query_prompt`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from sarb.config import parse_options
from sarb.contracts import EmbeddingKind, EmbeddingResponse, Vector
from sarb.dependencies import import_optional
from sarb.errors import EmbeddingError, SarbError
from sarb.providers.embedding._shared import (
    LocalEmbeddingOptions,
    ThreadRunner,
    check_output,
)

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from sarb.plugins.registry import ComponentContext

__all__ = ["SentenceTransformersOptions", "SentenceTransformersProvider", "build"]

PROVIDER = "sentence-transformers"
EXTRA = "sentence-transformers"


class SentenceTransformersOptions(LocalEmbeddingOptions):
    """Options for the Sentence Transformers adapter.

    Attributes:
        device: ``"cpu"``, ``"cuda"``, ``"mps"``. Unset means let torch decide.
        query_prompt: Prefix prepended to text embedded as a **query**.
        document_prompt: Prefix prepended to text embedded as a **document**.
        trust_remote_code: Execute model-authored Python from the Hub. See below.

    Why the two prompts are configuration:
        Asymmetric models each have their own convention and no way to announce it.
        E5 wants ``"query: "`` and ``"passage: "``. BGE wants an instruction on the
        query and nothing on the document. Many models want neither, and prefixing
        them makes retrieval *worse*.

        The framework cannot know which model you loaded, and a lookup table of model
        names to prompts would be wrong the week after it was written — so it asks.
        Both default to empty, which is the correct behaviour for a symmetric model
        and the wrong behaviour for an asymmetric one, silently. That is uncomfortable
        and it is still the honest option: the alternative is guessing, and a guess
        here costs recall that nobody will ever trace back to this line.

    Why ``trust_remote_code`` defaults to ``False``:
        It executes arbitrary Python downloaded from the Hub, in your process, with
        your credentials. Some genuinely good models need it. It is opt-in, per
        deployment, and it is spelled out in the config file rather than buried in a
        default — because someone should have to type it.
    """

    device: str | None = None
    query_prompt: str = ""
    document_prompt: str = ""
    trust_remote_code: bool = False

    # A torch forward pass on a CPU is memory-hungry. 32 is what the library itself
    # defaults to, and it is a sane place to start.
    batch_size: int = 32


class SentenceTransformersProvider:
    """The :class:`~sarb.contracts.EmbeddingProvider` port, backed by PyTorch.

    The loaded model is injected. :func:`build` loads it — a slow, network-touching
    operation that belongs at startup, not in the first request that happens to need
    a vector.
    """

    def __init__(self, model: Any, options: SentenceTransformersOptions) -> None:
        self._model = model
        self._options = options
        self._dimensions = int(model.get_sentence_embedding_dimension())
        self._runner = ThreadRunner(options.max_concurrency)

    @property
    def model(self) -> str:
        """The embedding model in use."""
        return self._options.model

    @property
    def dimensions(self) -> int:
        """The vector width, read from the loaded model itself."""
        return self._dimensions

    async def embed(
        self,
        texts: Sequence[str],
        *,
        kind: EmbeddingKind = EmbeddingKind.DOCUMENT,
    ) -> EmbeddingResponse:
        """Embed a batch of texts, off the event loop.

        No tokens are reported: the weights are in this process, and nobody invoices you
        for a forward pass on your own hardware.
        """
        if not texts:
            return EmbeddingResponse(vectors=[], model=self._options.model)

        prompt = (
            self._options.query_prompt
            if kind is EmbeddingKind.QUERY
            else self._options.document_prompt
        )
        batch = list(texts)

        def work() -> list[Vector]:
            # `normalize_embeddings` is delegated to the library rather than done in
            # our own Python: it is the same arithmetic, done in numpy, on a batch.
            vectors = self._model.encode(
                batch,
                prompt=prompt or None,
                batch_size=self._options.batch_size,
                normalize_embeddings=self._options.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [vector.tolist() for vector in vectors]

        try:
            vectors = await self._runner.run(work)
        except Exception as exc:
            raise self._fail(exc) from exc

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
        """Translate a torch failure into one of ours. None of them are retryable."""
        if isinstance(exc, SarbError):
            return exc

        return EmbeddingError(
            f"Sentence Transformers failed to embed: {exc}",
            provider=PROVIDER,
            context={"model": self._options.model},
            cause=exc,
        )


def build(ctx: ComponentContext) -> SentenceTransformersProvider:
    """Build the ``sentence-transformers`` embedder, loading the model now."""
    sdk = import_optional("sentence_transformers", extra=EXTRA, used_by=f"embedder {PROVIDER!r}")
    options = parse_options(ctx.config, SentenceTransformersOptions, component=ctx.path)

    model = sdk.SentenceTransformer(
        options.model,
        device=options.device,
        cache_folder=options.cache_dir,
        trust_remote_code=options.trust_remote_code,
    )
    return SentenceTransformersProvider(model, options)
