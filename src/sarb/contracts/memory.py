"""The memory port: conversation history.

Scoped narrowly, and on purpose
-------------------------------
"Memory" in agent frameworks is used for at least three different things:
conversation history, a rolling summary, and a semantic store of facts about a
user. They have genuinely different shapes — one is an append-only log, one is a
derived value, one is a search index — and a single port that covers all three
ends up as ``get(key)`` / ``set(key, value)``, which abstracts nothing.

So :class:`MemoryProvider` is the append-only log, only. Summarisation is a
*strategy layered on top* of it (a provider that summarises on write is a valid
implementation of this same port). Semantic recall is
:class:`~sarb.contracts.retrieval.RetrievalStrategy`, which already exists and
already does exactly that job.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from sarb.contracts.component import Family
from sarb.contracts.message import Message

__all__ = ["MEMORIES", "MemoryProvider"]


@runtime_checkable
class MemoryProvider(Protocol):
    """Stores and recalls a conversation.

    Implementations range from an in-process dict, through Redis, to Postgres. A
    provider that trims or summarises old turns on write is a legitimate
    implementation of this port and not a different one — which is the reason the
    port says nothing about *how much* it keeps.
    """

    async def append(self, session_id: str, messages: Sequence[Message]) -> None:
        """Append turns to a session's history.

        Batch, because a model turn plus its tool results arrive together and must
        land together — a history containing an assistant's tool call but not the
        results is one that no provider will accept on the next request.

        Raises:
            MemoryProviderError: The write failed.
        """
        ...

    async def history(self, session_id: str, *, limit: int | None = None) -> Sequence[Message]:
        """Recall a session's turns, oldest first.

        Args:
            session_id: The session. Unknown sessions return empty, not an error:
                the first turn of a conversation is a read of a session that does
                not exist yet, and that is not an exceptional case.
            limit: Return at most this many of the **most recent** turns. Note that
                a naive tail can slice between an assistant's tool call and its
                results; implementations must not return such a history.

        Raises:
            MemoryProviderError: The read failed.
        """
        ...

    async def clear(self, session_id: str) -> None:
        """Forget a session. Idempotent.

        Also the deletion primitive a "delete my data" request compiles down to, so
        it must actually delete rather than tombstone.
        """
        ...


#: The ``memories`` config family.
MEMORIES = Family("memories", MemoryProvider)
