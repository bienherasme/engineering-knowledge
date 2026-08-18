"""Persistence boundary: the port ingestion writes normalized knowledge through.

``Repository`` is deliberately small: one write operation and a handful of
read lookups, not a CRUD method per table. ``sync_source`` exists as a
single call, rather than separate create/update/delete methods plus a
transaction handle the caller has to manage, because one successful source
ingestion is meant to represent one atomic snapshot. Handing the adapter
the full incoming document set and letting it diff against what it already
has, inside one transaction, is what makes that atomicity a property of the
interface rather than something every caller has to get right by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from engineering_knowledge.domain import Chunk, Document, DocumentSource


class PersistenceError(Exception):
    """Base class for expected persistence-boundary failures."""


class UnsupportedSchemaVersionError(PersistenceError):
    """The database's schema version does not match what this code expects."""


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One chunk's embedding, as a persistence payload.

    Deliberately not part of ``Chunk``: a vector describes how our system
    represented a chunk for semantic search, not the chunk's own identity
    or content, the same reasoning that keeps ``processing_fingerprint`` off
    of ``Document``.
    """

    chunk_id: str
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    """The payload ingestion hands to a repository for one document.

    Not a normalized domain entity: it pairs a validated ``Document`` and
    its ``Chunk`` set with ``processing_fingerprint``, the piece of state
    that describes how our system produced them, which is deliberately not
    part of ``Document`` itself (see ``ingestion.processing`` for why).

    ``vectors``, ``embedding_fingerprint``, and ``embedding_dimension`` are
    all optional and travel together: ``None`` means this ingestion run had
    no embedding provider configured, and vector state is left untouched.
    When present, every ``ProcessedDocument`` in one ``sync_source`` call
    carries the same ``embedding_fingerprint``/``embedding_dimension``,
    since both describe the single active embedding configuration for that
    run, not anything document-specific.
    """

    document: Document
    chunks: tuple[Chunk, ...]
    processing_fingerprint: str
    vectors: tuple[VectorRecord, ...] | None = None
    embedding_fingerprint: str | None = None
    embedding_dimension: int | None = None


@dataclass(frozen=True, slots=True)
class SourceSyncResult:
    """Per-document outcome counts from one ``sync_source`` call.

    ``vector_reindexed`` counts documents whose vectors were written this
    call, independent of ``updated``/``reprocessed``: a document with
    unchanged content and unchanged processing can still be reindexed here
    if the active embedding configuration itself changed.
    """

    created: int
    updated: int
    reprocessed: int
    unchanged: int
    deleted: int
    vector_reindexed: int = 0


@dataclass(frozen=True, slots=True)
class VectorIndexingSummary:
    """Structured visibility into vector work, kept separate from IngestionResult.

    Not merged into the normalized created/updated/reprocessed/unchanged
    counts: those describe source synchronization, and vector reindexing is
    an independent axis that can happen without any of them changing.
    """

    embedding_fingerprint: str
    reindexed: int


class Repository(Protocol):
    """Persists normalized knowledge for one source at a time, atomically.

    ``sync_source`` treats ``processed_documents`` as the complete, current
    snapshot for ``source.source_id``: any previously persisted document
    for that source not present in this call is deleted along with its
    chunks. A caller that cannot fully discover and process a source should
    never call this with a partial list.
    """

    def sync_source(
        self, source: DocumentSource, processed_documents: Sequence[ProcessedDocument]
    ) -> SourceSyncResult: ...

    def get_document(self, document_id: str) -> Document | None: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...

    def get_chunks(self, document_id: str) -> tuple[Chunk, ...]: ...

    def list_documents_for_source(self, source_id: str) -> tuple[Document, ...]: ...
