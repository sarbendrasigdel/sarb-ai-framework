"""Document loading and chunking.

Chunk provenance
----------------
:class:`Chunk` carries ``start_char`` and ``end_char`` back into its source
document. That is not bookkeeping — it is what makes a citation *verifiable*. A
RAG system that can only say "this came from report.pdf" is asking the user to
trust it; one that can say "this came from characters 4120 to 4890 of report.pdf"
lets them check. Losing the offsets is easy and nearly impossible to add back
later, because by then a dozen chunkers have shipped without them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from sarb.contracts.base import Metadata, SarbModel, new_id
from sarb.contracts.component import Family

__all__ = [
    "CHUNKERS",
    "LOADERS",
    "Chunk",
    "ChunkingStrategy",
    "Document",
    "DocumentLoader",
    "DocumentSource",
]


class DocumentSource(SarbModel):
    """Where a document is being loaded from.

    One model covering a path, a URL, and bytes already in hand — because an upload
    arriving over HTTP is bytes, and a loader that can only take a path forces every
    API endpoint to write it to a temp file first.

    Attributes:
        uri: Path, URL, or other locator. Also the human-facing name of the source.
        data: The bytes, when the caller already has them.
        media_type: IANA type. Given explicitly, since sniffing is unreliable and
            an HTTP upload usually already knows.
        metadata: Attached to every document produced from this source. Where a
            tenant id, a permission tag, or an ingestion timestamp goes in.
    """

    uri: str
    data: bytes | None = None
    media_type: str | None = None
    metadata: Metadata = Field(default_factory=dict)


class Document(SarbModel):
    """A source document, extracted to text.

    Attributes:
        id: Stable identity. Re-ingesting the same document under the same id
            replaces it rather than duplicating it, so ``id`` should be derived from
            the source (a path, a content hash) and not freshly generated on each
            run — which is exactly the mistake the default invites, so derive it.
        content: The extracted text.
        source: Where it came from. Carried through to chunks and into citations.
        media_type: What it originally was.
        metadata: Application-defined. Vector-store filters match on these keys.
    """

    id: str = Field(default_factory=new_id)
    content: str
    source: str | None = None
    media_type: str | None = None
    metadata: Metadata = Field(default_factory=dict)


class Chunk(SarbModel):
    """A retrievable slice of a document.

    Attributes:
        id: Stable identity, used as the vector-store record id.
        document_id: The document it came from.
        text: The chunk's text — what gets embedded, and what the model reads.
        index: Position within the document, from zero. Lets neighbouring chunks be
            fetched to widen context at answer time.
        start_char: Offset of the chunk's first character in the source document.
        end_char: Offset one past its last character.
        source: Copied down from the document, so a hit can be cited without a
            second lookup.
        metadata: Copied down from the document, plus anything the chunker adds.
    """

    id: str = Field(default_factory=new_id)
    document_id: str
    text: str
    index: int = Field(ge=0)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    source: str | None = None
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _offsets_are_coherent(self) -> Self:
        """Half-open and ordered, or absent. A silently inverted span cites nonsense."""
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("Chunk needs both start_char and end_char, or neither")
        if (
            self.end_char is not None
            and self.start_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("Chunk end_char must not precede start_char")
        return self


@runtime_checkable
class DocumentLoader(Protocol):
    """Turns a source into documents.

    One source may yield many documents — a directory, a zip, a spreadsheet with a
    row per record.
    """

    def can_load(self, source: DocumentSource) -> bool:
        """Whether this loader handles this source.

        Lets a composite loader dispatch by type without a registry of file
        extensions in the engine.
        """
        ...

    def load(self, source: DocumentSource) -> AsyncIterator[Document]:
        """Load documents from a source.

        An async **iterator**, not a list. A loader pointed at a directory of ten
        thousand PDFs must not have to hold ten thousand extracted documents in
        memory before the first one can be embedded — and the list-returning version
        of this signature makes that impossible to fix later without changing every
        caller.

        Args:
            source: What to load.

        Yields:
            Documents, in whatever order the source presents them.

        Raises:
            DocumentError: The source could not be read or parsed.
        """
        ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Splits a document into retrievable pieces.

    Async because good chunking is not always cheap or local: semantic chunking
    calls an embedding model, and layout-aware chunking may call a service. A sync
    signature would force those to block the event loop, or to lie.
    """

    async def chunk(self, document: Document) -> Sequence[Chunk]:
        """Split one document.

        Implementations must set ``document_id``, ``index``, and the character
        offsets. See the module docstring on why the offsets are not optional in
        spirit even though they are in type.

        Args:
            document: The document to split.

        Returns:
            Chunks in document order.

        Raises:
            ChunkingError: The document could not be split.
        """
        ...


#: The ``loaders`` config family.
LOADERS = Family("loaders", DocumentLoader)

#: The ``chunkers`` config family.
CHUNKERS = Family("chunkers", ChunkingStrategy)
