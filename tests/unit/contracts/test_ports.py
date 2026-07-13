"""Conformance tests for the eleven ports.

Two things are asserted, and both are structural rather than behavioural:

1. **A plain class satisfies a port without importing anything from sarb.** This
   is the property that makes the Protocol choice worth its weight — an adapter
   can be written with zero framework imports, and a downstream team can satisfy a
   port with a class they already have. If a port ever grows a requirement that
   breaks it, this fails.

2. **Every port has a family, and every family is configurable.** A port whose
   family is missing from `Settings` is a port nobody can wire up.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from sarb.config import load_settings
from sarb.contracts import (
    CHUNKERS,
    EMBEDDERS,
    LLMS,
    LOADERS,
    MEMORIES,
    PROMPTS,
    RETRIEVERS,
    STORAGES,
    TOOLS,
    VECTOR_STORES,
    WORKFLOWS,
    Chunk,
    ChunkingStrategy,
    Distance,
    Document,
    DocumentLoader,
    DocumentSource,
    EmbeddingKind,
    EmbeddingProvider,
    Family,
    Filter,
    LLMCapabilities,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MemoryProvider,
    Message,
    PromptProvider,
    PromptTemplate,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    SearchHit,
    StorageProvider,
    StoredObject,
    StreamEnd,
    StreamEvent,
    ToolCall,
    ToolContext,
    ToolProvider,
    ToolResult,
    ToolSpec,
    Vector,
    VectorQuery,
    VectorRecord,
    VectorStore,
    Workflow,
    WorkflowEvent,
    WorkflowExecutor,
    WorkflowFinished,
    WorkflowRun,
)

ALL_FAMILIES = [
    LLMS,
    EMBEDDERS,
    VECTOR_STORES,
    MEMORIES,
    PROMPTS,
    LOADERS,
    CHUNKERS,
    RETRIEVERS,
    TOOLS,
    WORKFLOWS,
    STORAGES,
]


# --------------------------------------------------------------------------- #
# Minimal implementations. Note what is absent from every one of them: any
# inheritance, and any import of a sarb base class. That is the whole point.
# --------------------------------------------------------------------------- #


class FakeLLM:
    @property
    def model(self) -> str:
        return "fake"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(tools=True)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(message=Message.assistant("ok"), model="fake")

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEnd(response=await self.generate(request))


class FakeEmbedder:
    @property
    def model(self) -> str:
        return "fake"

    @property
    def dimensions(self) -> int:
        return 3

    async def embed(
        self, texts: Sequence[str], *, kind: EmbeddingKind = EmbeddingKind.DOCUMENT
    ) -> list[Vector]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    async def ensure_collection(
        self, collection: str, *, dimensions: int, distance: Distance = Distance.COSINE
    ) -> None: ...

    async def drop_collection(self, collection: str) -> None: ...

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None: ...

    async def search(self, collection: str, query: VectorQuery) -> Sequence[SearchHit]:
        return [SearchHit(id="a", score=0.9)]

    async def fetch(self, collection: str, ids: Sequence[str]) -> Sequence[VectorRecord]:
        return []

    async def delete(
        self,
        collection: str,
        *,
        ids: Sequence[str] | None = None,
        where: Filter | None = None,
    ) -> int:
        return 0

    async def count(self, collection: str, *, where: Filter | None = None) -> int:
        return 0


class FakeMemory:
    async def append(self, session_id: str, messages: Sequence[Message]) -> None: ...

    async def history(self, session_id: str, *, limit: int | None = None) -> Sequence[Message]:
        return []

    async def clear(self, session_id: str) -> None: ...


class FakePrompts:
    async def get(self, name: str, *, version: str | None = None) -> PromptTemplate:
        return PromptTemplate(name=name, template="hi")

    async def list(self) -> Sequence[PromptTemplate]:
        return []


class FakeLoader:
    def can_load(self, source: DocumentSource) -> bool:
        return True

    async def load(self, source: DocumentSource) -> AsyncIterator[Document]:
        yield Document(content="hello", source=source.uri)


class FakeChunker:
    async def chunk(self, document: Document) -> Sequence[Chunk]:
        return [
            Chunk(
                document_id=document.id,
                text=document.content,
                index=0,
                start_char=0,
                end_char=len(document.content),
            )
        ]


class FakeRetriever:
    async def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalResult]:
        return [RetrievalResult(chunk=Chunk(document_id="d1", text="Paris", index=0), score=0.9)]


class FakeTools:
    async def list_tools(self) -> Sequence[ToolSpec]:
        return [ToolSpec(name="search", description="Search.")]

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        return ToolResult(call_id=call.id, name=call.name, content="ok")


class FakeWorkflows:
    async def execute(self, workflow: Workflow, inputs: dict[str, Any]) -> WorkflowRun:
        return WorkflowRun(workflow=workflow.name)

    async def stream(
        self, workflow: Workflow, inputs: dict[str, Any]
    ) -> AsyncIterator[WorkflowEvent]:
        yield WorkflowFinished(run=await self.execute(workflow, inputs))


class FakeStorage:
    async def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredObject:
        return StoredObject(key=key, size=len(data))

    async def get(self, key: str) -> bytes:
        return b""

    async def stream(self, key: str) -> AsyncIterator[bytes]:
        yield b""

    async def delete(self, key: str) -> None: ...

    async def exists(self, key: str) -> bool:
        return False

    async def list(self, prefix: str = "") -> AsyncIterator[StoredObject]:
        yield StoredObject(key="a", size=0)

    async def url(self, key: str, *, expires_in: int = 3600) -> str:
        return "http://x"


IMPLEMENTATIONS = [
    (FakeLLM(), LLMProvider),
    (FakeEmbedder(), EmbeddingProvider),
    (FakeVectorStore(), VectorStore),
    (FakeMemory(), MemoryProvider),
    (FakePrompts(), PromptProvider),
    (FakeLoader(), DocumentLoader),
    (FakeChunker(), ChunkingStrategy),
    (FakeRetriever(), RetrievalStrategy),
    (FakeTools(), ToolProvider),
    (FakeWorkflows(), WorkflowExecutor),
    (FakeStorage(), StorageProvider),
]


@pytest.mark.parametrize(
    ("implementation", "port"),
    IMPLEMENTATIONS,
    ids=[port.__name__ for _, port in IMPLEMENTATIONS],
)
def test_a_plain_class_satisfies_its_port(implementation: object, port: type) -> None:
    """No inheritance, no sarb import — just the right shape."""
    assert isinstance(implementation, port)


def test_a_class_of_the_wrong_shape_does_not_satisfy_a_port() -> None:
    """Guards against the ports being so loose that anything passes."""

    class NotAnLLM:
        def greet(self) -> str:
            return "hi"

    assert not isinstance(NotAnLLM(), LLMProvider)


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
def test_every_port_has_a_configurable_family(family: Family[Any]) -> None:
    """A port whose family is not in Settings is a port nobody can wire up."""
    settings = load_settings()

    assert isinstance(getattr(settings, family.name), dict)


def test_there_are_eleven_families_one_per_required_abstraction() -> None:
    assert len({family.name for family in ALL_FAMILIES}) == 11
