"""Ports and models — the innermost ring.

Everything here is depended upon and depends on nothing: no vendor SDK, no
configuration, no container. Engines are written against these types, and adapters
implement them.

The eleven ports, and the config family each is resolved from:

| Port | Family | What it abstracts |
| --- | --- | --- |
| `LLMProvider` | `llms` | language models |
| `EmbeddingProvider` | `embedders` | text → vectors |
| `VectorStore` | `vector_stores` | vector storage and search |
| `MemoryProvider` | `memories` | conversation history |
| `PromptProvider` | `prompts` | versioned prompts |
| `DocumentLoader` | `loaders` | source → documents |
| `ChunkingStrategy` | `chunkers` | document → chunks |
| `RetrievalStrategy` | `retrievers` | question → passages |
| `ToolProvider` | `tools` | callable tools |
| `WorkflowExecutor` | `workflows` | step orchestration |
| `StorageProvider` | `storages` | blobs |

Ports are :class:`typing.Protocol`, not abstract base classes. Structural typing
means a class satisfies a port by *shape*, without importing or subclassing
anything from ``sarb`` — the strongest available form of dependency inversion, and
the difference between "depend on our abstraction" and "depend on a shape".
"""

from sarb.contracts.base import JSONValue, Metadata, SarbModel, new_id
from sarb.contracts.component import (
    DEFAULT_COMPONENT,
    Closable,
    ComponentResolver,
    Family,
    Scope,
)
from sarb.contracts.document import (
    CHUNKERS,
    LOADERS,
    Chunk,
    ChunkingStrategy,
    Document,
    DocumentLoader,
    DocumentSource,
)
from sarb.contracts.embedding import (
    EMBEDDERS,
    EmbeddingKind,
    EmbeddingProvider,
    EmbeddingResponse,
    Vector,
)
from sarb.contracts.llm import (
    LLMS,
    LLMCapabilities,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ReasoningDelta,
    ResponseFormat,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextDelta,
    TokenCounter,
    ToolCallDelta,
    ToolChoice,
)
from sarb.contracts.memory import MEMORIES, MemoryProvider
from sarb.contracts.message import (
    ContentPart,
    FinishReason,
    ImagePart,
    Message,
    ReasoningPart,
    Role,
    TextPart,
    ToolCall,
    ToolResult,
    Usage,
)
from sarb.contracts.observability import Logger, Meter, Span, Tracer
from sarb.contracts.prompt import PROMPTS, PromptProvider, PromptTemplate
from sarb.contracts.retrieval import (
    RETRIEVERS,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
)
from sarb.contracts.storage import STORAGES, StorageProvider, StoredObject
from sarb.contracts.tool import TOOLS, ToolContext, ToolProvider, ToolSpec
from sarb.contracts.vector_store import (
    VECTOR_STORES,
    And,
    Compare,
    CompareOp,
    Distance,
    Filter,
    Not,
    Or,
    SearchHit,
    VectorQuery,
    VectorRecord,
    VectorStore,
    eq,
    exists,
    gt,
    gte,
    in_,
    lt,
    lte,
    ne,
    not_in,
)
from sarb.contracts.workflow import (
    WORKFLOWS,
    RunStatus,
    Step,
    StepFinished,
    StepStarted,
    Workflow,
    WorkflowEvent,
    WorkflowExecutor,
    WorkflowFinished,
    WorkflowRun,
)

__all__ = [
    "CHUNKERS",
    "DEFAULT_COMPONENT",
    "EMBEDDERS",
    "LLMS",
    "LOADERS",
    "MEMORIES",
    "PROMPTS",
    "RETRIEVERS",
    "STORAGES",
    "TOOLS",
    "VECTOR_STORES",
    "WORKFLOWS",
    "And",
    "Chunk",
    "ChunkingStrategy",
    "Closable",
    "Compare",
    "CompareOp",
    "ComponentResolver",
    "ContentPart",
    "Distance",
    "Document",
    "DocumentLoader",
    "DocumentSource",
    "EmbeddingKind",
    "EmbeddingProvider",
    "EmbeddingResponse",
    "Family",
    "Filter",
    "FinishReason",
    "ImagePart",
    "JSONValue",
    "LLMCapabilities",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "Logger",
    "MemoryProvider",
    "Message",
    "Metadata",
    "Meter",
    "Not",
    "Or",
    "PromptProvider",
    "PromptTemplate",
    "ReasoningDelta",
    "ReasoningPart",
    "ResponseFormat",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalStrategy",
    "Role",
    "RunStatus",
    "SarbModel",
    "Scope",
    "SearchHit",
    "Span",
    "Step",
    "StepFinished",
    "StepStarted",
    "StorageProvider",
    "StoredObject",
    "StreamEnd",
    "StreamEvent",
    "StreamStart",
    "TextDelta",
    "TextPart",
    "TokenCounter",
    "ToolCall",
    "ToolCallDelta",
    "ToolChoice",
    "ToolContext",
    "ToolProvider",
    "ToolResult",
    "ToolSpec",
    "Tracer",
    "Usage",
    "Vector",
    "VectorQuery",
    "VectorRecord",
    "VectorStore",
    "Workflow",
    "WorkflowEvent",
    "WorkflowExecutor",
    "WorkflowFinished",
    "WorkflowRun",
    "eq",
    "exists",
    "gt",
    "gte",
    "in_",
    "lt",
    "lte",
    "ne",
    "new_id",
    "not_in",
]
