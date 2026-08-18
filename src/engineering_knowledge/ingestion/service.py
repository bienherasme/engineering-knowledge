"""Source ingestion orchestration.

One call to ``ingest_source`` represents one attempt to bring persisted
state for a single configured source fully in line with what that source
currently contains. Discovery and processing happen entirely in memory
first, by fully materializing the adapter's document iterator, before any
repository call is made: a failure partway through discovery or processing
must leave previously persisted state untouched, and the only way to
guarantee that here is to never start writing until everything to write is
already known to be valid. When an embedding provider is configured, the
same applies to embedding: every vector is computed in memory before
``sync_source`` opens its transaction, so a model failure never touches the
database either.

Every document is normalized and chunked in memory on every ingestion run,
including ones that turn out to be unchanged. That is deliberate: for the
small local corpus this system targets, recomputing cheap, in-process
chunking is far simpler than trying to short-circuit it based on state this
service does not own, and it changes nothing about what actually reaches
SQLite. The repository is the only thing that decides whether an unchanged
document gets written; ``sync_source`` never rewrites a row whose content
and processing fingerprint both match what is already persisted, and the
same is true of vectors against the active embedding fingerprint.

``embedding_provider`` is optional and defaults to ``None``: base ingestion
never imports or requires anything vector-related. Nothing in this module
imports sqlite-vec or sentence-transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engineering_knowledge.embeddings.base import EmbeddingProvider
from engineering_knowledge.embeddings.fingerprint import (
    build_embedding_text,
    derive_embedding_fingerprint,
)
from engineering_knowledge.ingestion.processing import process_raw_document
from engineering_knowledge.persistence.base import ProcessedDocument, Repository, VectorRecord
from engineering_knowledge.sources.base import SourceAdapter


class VectorIndexingSummary(BaseModel):
    """Structured visibility into vector work from one ingestion run.

    Kept separate from the normalized created/updated/reprocessed/unchanged
    counts on ``IngestionResult``: vector reindexing is an independent axis
    that can happen without any of those counts changing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding_fingerprint: str
    reindexed: int = Field(ge=0)


class IngestionResult(BaseModel):
    """Structured summary of one successful source ingestion.

    ``discovered`` counts documents seen in this run, not chunks.
    ``deleted`` is intentionally separate from the discovered total: a
    deleted document was, by definition, not part of this run's discovery.
    ``vector_indexing`` is ``None`` whenever no embedding provider was
    configured for this run; its presence never affects the arithmetic
    invariant below, which describes normalized source synchronization only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    discovered: int = Field(ge=0)
    created: int = Field(ge=0)
    updated: int = Field(ge=0)
    reprocessed: int = Field(ge=0)
    unchanged: int = Field(ge=0)
    deleted: int = Field(ge=0)
    vector_indexing: VectorIndexingSummary | None = None

    @model_validator(mode="after")
    def _check_discovered_total(self) -> IngestionResult:
        expected = self.created + self.updated + self.reprocessed + self.unchanged
        if self.discovered != expected:
            raise ValueError("discovered must equal created + updated + reprocessed + unchanged")
        return self


class IngestionService:
    """Ties a SourceAdapter to a Repository under one processing configuration.

    ``max_chunk_chars`` is processing configuration, not per-call input: it
    participates in every processed document's processing_fingerprint, so
    constructing a service with a different bound and re-ingesting the same
    source is exactly what causes previously persisted documents to be
    reported as reprocessed.

    ``embedding_provider``, when given, participates in a completely
    separate fingerprint (``embedding_fingerprint``): changing the provider
    or model changes that fingerprint, which causes vectors to be rebuilt,
    but never changes ``processing_fingerprint`` and never causes a document
    to be reported as updated or reprocessed. Processing and embedding are
    independent invalidation axes by design.
    """

    def __init__(
        self,
        repository: Repository,
        *,
        max_chunk_chars: int,
        embedding_provider: EmbeddingProvider | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._max_chunk_chars = max_chunk_chars
        self._embedding_provider = embedding_provider
        self._clock = clock

    def ingest_source(self, adapter: SourceAdapter) -> IngestionResult:
        source = adapter.source()
        raw_documents = list(adapter.discover())

        now = self._clock()
        processed_documents = [
            process_raw_document(raw, max_chunk_chars=self._max_chunk_chars, now=now)
            for raw in raw_documents
        ]

        embedding_fingerprint = None
        if self._embedding_provider is not None:
            processed_documents, embedding_fingerprint = self._attach_vectors(
                processed_documents, self._embedding_provider
            )

        sync_result = self._repository.sync_source(source, processed_documents)

        vector_indexing = None
        if embedding_fingerprint is not None:
            vector_indexing = VectorIndexingSummary(
                embedding_fingerprint=embedding_fingerprint,
                reindexed=sync_result.vector_reindexed,
            )

        return IngestionResult(
            source_id=source.source_id,
            discovered=len(processed_documents),
            created=sync_result.created,
            updated=sync_result.updated,
            reprocessed=sync_result.reprocessed,
            unchanged=sync_result.unchanged,
            deleted=sync_result.deleted,
            vector_indexing=vector_indexing,
        )

    @staticmethod
    def _attach_vectors(
        processed_documents: list[ProcessedDocument], provider: EmbeddingProvider
    ) -> tuple[list[ProcessedDocument], str]:
        # Embedding text is built per chunk and every chunk across every
        # document is embedded in one batched call: cheaper than one call
        # per document, and this still happens entirely before sync_source
        # opens its transaction.
        texts = [
            build_embedding_text(
                title=processed.document.title, section_path=chunk.section_path,
                chunk_text=chunk.text,
            )
            for processed in processed_documents
            for chunk in processed.chunks
        ]
        vectors = provider.embed_documents(texts) if texts else ()
        embedding_fingerprint = derive_embedding_fingerprint(
            provider_type=provider.provider_type,
            model_id=provider.model_id,
            model_revision=provider.model_revision,
            dimension=provider.dimension,
        )

        vector_iterator = iter(vectors)
        updated_documents = []
        for processed in processed_documents:
            document_vectors = tuple(
                VectorRecord(chunk_id=chunk.chunk_id, embedding=next(vector_iterator))
                for chunk in processed.chunks
            )
            updated_documents.append(
                replace(
                    processed,
                    vectors=document_vectors,
                    embedding_fingerprint=embedding_fingerprint,
                    embedding_dimension=provider.dimension,
                )
            )
        return updated_documents, embedding_fingerprint
