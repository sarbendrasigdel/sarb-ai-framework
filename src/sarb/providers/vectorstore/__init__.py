"""Vector-store adapters.

| ``provider:`` | Extra | What it is |
| --- | --- | --- |
| ``qdrant`` | ``sarb[qdrant]`` | Qdrant. Purpose-built, and the fastest of the three |
| ``pgvector`` | ``sarb[pgvector]`` | The Postgres you already run |
| ``memory`` | — | Exhaustive search, in this process. No server |

The conventions are the ones the other provider families follow — the SDK is imported
lazily inside the factory, and the client is injected rather than constructed — plus one
that belongs to this family alone:

**A filter that cannot be expressed must raise.** Never dropped, never approximated. A
dropped filter returns *more* rows, not fewer: it fails **open**. A ``tenant`` filter
that is silently ignored does not error and does not return nothing — it returns another
tenant's documents, ranked by relevance, to whoever asked.

That rule is why :mod:`sarb.providers.vectorstore.memory` is a real, exhaustive store
rather than a stub. It is the reference implementation of the filter algebra, and the
conformance suite runs the same filters against every backend and demands the same
answers — which is how Qdrant's ``ne`` was found to be quietly matching records that had
no such field at all.
"""

from __future__ import annotations

from sarb.providers.vectorstore.memory import MemoryVectorStore, MemoryVectorStoreOptions
from sarb.providers.vectorstore.pgvector import PgVectorOptions, PgVectorStore
from sarb.providers.vectorstore.qdrant import QdrantOptions, QdrantStore

__all__ = [
    "MemoryVectorStore",
    "MemoryVectorStoreOptions",
    "PgVectorOptions",
    "PgVectorStore",
    "QdrantOptions",
    "QdrantStore",
]
