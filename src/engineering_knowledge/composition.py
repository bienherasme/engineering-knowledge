"""Small explicit factory functions wiring typed configuration to services.

Not a DI framework, service locator, or global container: each function
takes exactly the inputs it needs and returns one constructed object. The
CLI and MCP adapters both compose from these instead of each reimplementing
how a ``KnowledgeService`` or an ``IngestionService`` gets built.

Composition is capability-specific and lazy on purpose: a caller that only
needs lexical search never has to construct an embedding provider, and a
caller building a read-only knowledge-serving service never gets a writable
repository. Nothing here decides *which* capability a command needs; each
factory is called only when that capability is actually required.
"""

from __future__ import annotations

from engineering_knowledge.config import AppConfig, ConfigurationError
from engineering_knowledge.embeddings.base import EmbeddingProvider
from engineering_knowledge.embeddings.sentence_transformers_provider import (
    SentenceTransformersEmbeddingProvider,
)
from engineering_knowledge.ingestion.service import IngestionService
from engineering_knowledge.persistence.base import Repository
from engineering_knowledge.persistence.sqlite import SqliteRepository
from engineering_knowledge.retrieval.service import KnowledgeService
from engineering_knowledge.retrieval.vector import VectorRetriever
from engineering_knowledge.sources.local_filesystem import LocalFilesystemSourceAdapter


def open_repository(
    config: AppConfig, *, vector_index_enabled: bool = False, read_only: bool = False
) -> SqliteRepository:
    """Open the configured database, writable by default.

    A writable open creates the database's parent directory if missing,
    the normal first-run convenience for a local maintenance command. A
    read-only open never touches the filesystem beyond opening the
    existing file: knowledge-serving composition must not be able to
    create, migrate, or otherwise prepare a database.
    """
    db_path = config.persistence.db_path
    if read_only:
        return SqliteRepository(
            str(db_path), vector_index_enabled=vector_index_enabled, read_only=True
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteRepository(str(db_path), vector_index_enabled=vector_index_enabled)


def build_source_adapter(config: AppConfig) -> LocalFilesystemSourceAdapter:
    return LocalFilesystemSourceAdapter(
        source_id=config.source.source_id,
        root=config.source.root,
        max_file_size_bytes=config.source.max_file_size_bytes,
    )


def build_ingestion_service(
    config: AppConfig,
    repository: Repository,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> IngestionService:
    return IngestionService(
        repository,
        max_chunk_chars=config.processing.max_chunk_chars,
        embedding_provider=embedding_provider,
    )


def build_embedding_provider(config: AppConfig) -> EmbeddingProvider:
    """Construct the configured real embedding provider.

    ``config.embeddings.provider`` is already validated to be a supported
    value by the time an ``AppConfig`` exists, so the only thing left to
    check here is whether embeddings are enabled at all: a caller that
    needs vector capability but configured ``enabled = false`` gets a
    ``ConfigurationError``, not a silently degraded lexical-only path.
    Missing the optional ``local-embeddings`` dependency surfaces as
    ``EmbeddingConfigurationError`` from the provider constructor itself.
    """
    if not config.embeddings.enabled:
        raise ConfigurationError(
            "embeddings.enabled is false in configuration; this operation requires "
            "configured vector capability"
        )
    return SentenceTransformersEmbeddingProvider(model_id=config.embeddings.model_id)


def build_vector_retriever(
    embedding_provider: EmbeddingProvider, repository: SqliteRepository
) -> VectorRetriever:
    return VectorRetriever(embedding_provider, repository)


def build_knowledge_service(
    config: AppConfig,
    repository: SqliteRepository,
    *,
    vector_retriever: VectorRetriever | None = None,
) -> KnowledgeService:
    return KnowledgeService(
        repository,
        repository,
        vector_retriever,
        default_strategy=config.retrieval.default_strategy,
    )
