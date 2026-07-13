"""The storage port: bytes, by key.

Object storage — the original PDF behind a document, an uploaded image, an
exported evaluation report. Local disk in development, S3 or GCS in production,
same code.

Deliberately *not* a general key-value store. Redis-shaped caching and blob-shaped
storage have different access patterns, different size assumptions, and different
durability guarantees; a port that spans both would be ``get``/``set`` and would
abstract nothing useful.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import Field

from sarb.contracts.base import Metadata, SarbModel
from sarb.contracts.component import Family

__all__ = ["STORAGES", "StorageProvider", "StoredObject"]


class StoredObject(SarbModel):
    """Metadata about a stored blob — never its bytes.

    Listing a bucket must not download it, so the listing type cannot carry content.
    """

    key: str
    size: int = Field(ge=0)
    content_type: str | None = None
    modified_at: datetime | None = None
    metadata: Metadata = Field(default_factory=dict)


@runtime_checkable
class StorageProvider(Protocol):
    """Stores and retrieves blobs by key."""

    async def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: Metadata | None = None,
    ) -> StoredObject:
        """Write a blob, replacing any existing one at the key.

        Raises:
            StorageError: The write failed.
        """
        ...

    async def get(self, key: str) -> bytes:
        """Read a whole blob.

        Raises:
            StorageError: No such key, or the read failed.
        """
        ...

    def stream(self, key: str) -> AsyncIterator[bytes]:
        """Read a blob in chunks.

        The reason this exists next to :meth:`get`: a 2 GB video does not fit in the
        memory of a container sized for a web service. Whenever the size is not
        known to be small, stream.

        Raises:
            StorageError: No such key, or the read failed.
        """
        ...

    async def delete(self, key: str) -> None:
        """Delete a blob. Idempotent — deleting an absent key is not an error."""
        ...

    async def exists(self, key: str) -> bool:
        """Whether a key is present."""
        ...

    def list(self, prefix: str = "") -> AsyncIterator[StoredObject]:
        """List objects under a prefix.

        An async iterator, because a bucket can hold millions of objects and every
        real backend paginates. A list-returning signature makes that pagination the
        caller's problem, and the caller always forgets.
        """
        ...

    async def url(self, key: str, *, expires_in: int = 3600) -> str:
        """A time-limited URL a browser can fetch directly.

        Lets a client download a 500 MB export without it passing through the
        application at all.

        Args:
            key: The object.
            expires_in: Lifetime in seconds.

        Raises:
            StorageError: The backend cannot issue URLs, or the object is absent.
        """
        ...


#: The ``storages`` config family.
STORAGES = Family("storages", StorageProvider)
