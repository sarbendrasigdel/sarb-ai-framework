"""A deterministic embedder. No model, no network, no key.

Like the fake LLM, this is a **registered provider**, not a test fixture: put
``provider: fake`` in the ``embedders`` block and the whole application — indexing,
retrieval, the API — runs on a laptop with nothing installed.

Its vectors are derived from a hash of the text, so the same text always produces the
same vector, in this process and every future one. That is what makes it usable for
retrieval tests: the geometry is stable, so an assertion about *which* chunk came back
first is an assertion about the retriever, not about the weather.

It also records how each text was embedded, which turns the nastiest bug in this
module — embedding a query as a document — from silent into visible. See
:attr:`FakeEmbeddingProvider.kinds`.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from sarb.config import parse_options
from sarb.contracts import EmbeddingKind, EmbeddingResponse, Usage, Vector
from sarb.errors import AuthenticationError, EmbeddingError, ProviderTimeoutError, RateLimitError
from sarb.providers.embedding._shared import EmbeddingOptions, check_output, normalized

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from sarb.plugins.registry import ComponentContext

__all__ = ["FakeEmbeddingOptions", "FakeEmbeddingProvider", "build"]

PROVIDER = "fake"

type Failure = Literal["rate_limit", "auth", "timeout", "error"]


class FakeEmbeddingOptions(EmbeddingOptions):
    """Options for the fake embedder.

    Attributes:
        model: Anything. It is recorded and reported, and nothing loads it.
        dimensions: The vector width. Small by default — eight floats are as good as
            a thousand for asserting that plumbing works, and far easier to read in a
            failing test.
        vectors: Text to exact vector. For tests that need to *place* things.
        fail_with: Fail instead of embedding, in a chosen way.
        fail_times: How many calls fail before it starts working. ``None`` means every
            call, forever — which cannot express the case a retry policy exists for:
            rate limited twice, then fine.

    Why ``vectors`` is worth having:
        Hashed vectors are deterministic but arbitrary: you cannot ask for "a document
        that is close to this query" without knowing where the hash happened to put
        it. Scripting the geometry directly lets a retrieval test say what it means —
        these two chunks are near the query, that one is far — and then assert the
        ranking that must follow. Anything not listed still falls back to the hash.
    """

    model: str = "fake"
    dimensions: int = Field(default=8, gt=0)
    vectors: dict[str, tuple[float, ...]] = Field(default_factory=dict)
    fail_with: Failure | None = None
    fail_times: int | None = Field(default=None, ge=0)


class FakeEmbeddingProvider:
    """The :class:`~sarb.contracts.EmbeddingProvider` port, with nothing behind it.

    Attributes:
        texts: Every text embedded, in order.
        kinds: The :class:`~sarb.contracts.EmbeddingKind` each one was embedded as.
    """

    def __init__(self, options: FakeEmbeddingOptions) -> None:
        self._options = options
        self._failures = 0
        self.texts: list[str] = []
        self.kinds: list[EmbeddingKind] = []

    @property
    def model(self) -> str:
        """The configured model name."""
        return self._options.model

    @property
    def dimensions(self) -> int:
        """The configured vector width."""
        return self._options.dimensions

    async def embed(
        self,
        texts: Sequence[str],
        *,
        kind: EmbeddingKind = EmbeddingKind.DOCUMENT,
    ) -> EmbeddingResponse:
        """Return a deterministic vector for each text.

        Recording ``kind`` is the point of this method as much as the vectors are. An
        indexing pipeline that embeds queries as documents, or a retriever that embeds
        its query as a passage, produces a system that works — just worse — and no
        test that only looks at the vectors will ever catch it. One that looks at
        :attr:`kinds` catches it immediately.
        """
        self.texts.extend(texts)
        self.kinds.extend([kind] * len(texts))

        if self._options.fail_with and self._failing():
            raise self._failure()

        model = self._options.model

        if not texts:
            return EmbeddingResponse(vectors=[], model=model)

        vectors = [self._vector(text) for text in texts]

        return EmbeddingResponse(
            vectors=check_output(
                vectors,
                texts,
                provider=PROVIDER,
                model=model,
                dimensions=self._options.dimensions,
            ),
            model=model,
            # A crude but plausible count, on the same four-characters-a-token rule the
            # fake LLM uses. Zero would be more honest about the fake and much less
            # useful: cost and budgeting code needs something to divide by, and a test
            # of it should not need a real API key.
            usage=Usage(input_tokens=sum(len(text) for text in texts) // 4),
        )

    def _failing(self) -> bool:
        """Whether this call is one of the scripted failures."""
        if self._options.fail_times is None:
            return True

        if self._failures >= self._options.fail_times:
            return False

        self._failures += 1
        return True

    def _vector(self, text: str) -> Vector:
        """A scripted vector if there is one, otherwise a hashed one.

        Seeded from a SHA-256 of the text rather than :func:`hash`, whose salt changes
        between processes — a "deterministic" fake that gives different answers on the
        next run is worse than an obviously random one, because it makes a flaky test
        look like a real failure.
        """
        scripted = self._options.vectors.get(text)
        if scripted is not None:
            return normalized(list(scripted))

        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        return normalized([rng.uniform(-1.0, 1.0) for _ in range(self._options.dimensions)])

    def _failure(self) -> Exception:
        """The scripted failure. The only honest way to test a retry policy."""
        match self._options.fail_with:
            case "rate_limit":
                return RateLimitError("scripted rate limit", retry_after=1.0, provider=PROVIDER)
            case "auth":
                return AuthenticationError("scripted auth failure", provider=PROVIDER)
            case "timeout":
                return ProviderTimeoutError("scripted timeout", provider=PROVIDER)
            case _:
                return EmbeddingError("scripted failure", provider=PROVIDER)


def build(ctx: ComponentContext) -> FakeEmbeddingProvider:
    """Build the ``fake`` embedder from configuration."""
    return FakeEmbeddingProvider(
        parse_options(ctx.config, FakeEmbeddingOptions, component=ctx.path)
    )
