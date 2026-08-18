"""MCP stdio adapter: read-only knowledge access, nothing else.

This is the only module in the codebase that imports the ``mcp`` SDK.
Exactly three tools are exposed: ``search_knowledge``, ``get_document``,
``get_chunk``. There is no ingest, rebuild, evaluate, filesystem, SQL, or
mutation tool, and there never should be: an MCP client gets retrieval
evidence over an already-prepared database, not maintenance capability or
a generated answer.

``run_stdio_server`` composes its ``KnowledgeService`` from a read-only
repository (see ``persistence.sqlite``'s ``read_only`` mode): serving never
ingests a source, migrates a schema, or rebuilds a vector index. Those stay
explicit CLI operations.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ConfigDict

from engineering_knowledge.composition import (
    build_embedding_provider,
    build_knowledge_service,
    build_vector_retriever,
    open_repository,
)
from engineering_knowledge.config import AppConfig
from engineering_knowledge.domain import Chunk, Document
from engineering_knowledge.persistence.base import PersistenceError
from engineering_knowledge.retrieval.errors import (
    IndexUnavailableError,
    InvalidQueryError,
    VectorIndexIncompatibleError,
    VectorIndexUnavailableError,
)
from engineering_knowledge.retrieval.service import (
    DEFAULT_PUBLIC_MAX_RESULTS,
    KnowledgeService,
    RetrievalResult,
    RetrievalStrategy,
)

SERVER_NAME = "engineering-knowledge"

# Every one of these is an expected application-boundary outcome (a bad
# query, an unavailable or incompatible vector index, a persistence read
# failure): sanitized into a ToolError with its own concise message.
# Anything else is a programming bug and is left to propagate; the SDK's
# own call_tool handler still turns it into a non-crashing tool error
# result, it just is not relabeled as one of these known cases.
_EXPECTED_ERRORS = (
    InvalidQueryError,
    IndexUnavailableError,
    VectorIndexUnavailableError,
    VectorIndexIncompatibleError,
    PersistenceError,
)


class DocumentLookupResult(BaseModel):
    """A normal identity-lookup outcome, not an infrastructure failure.

    ``found=False`` means no document exists for the given id; it is never
    represented as an empty/fake ``Document``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    found: bool
    document: Document | None = None


class ChunkLookupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    found: bool
    chunk: Chunk | None = None


def build_server(knowledge_service: KnowledgeService) -> MCPServer:
    server: MCPServer = MCPServer(
        name=SERVER_NAME,
        instructions=(
            "Read-only access to a pre-built local engineering knowledge base. "
            "Tools return retrieval evidence with provenance, never a generated answer."
        ),
    )

    # Tool bodies are synchronous SQLite reads over one connection created
    # on this thread. Declared `async def` deliberately, not because they
    # await anything, but because the SDK offloads plain `def` tools to a
    # worker thread pool (anyio.to_thread.run_sync) to avoid blocking the
    # event loop on slow synchronous work; a sqlite3 connection is bound to
    # the thread that created it, so running it from a worker thread would
    # break outright. `async def` keeps the call on this thread instead.

    @server.tool()
    async def search_knowledge(
        query: str,
        strategy: str | None = None,
        max_results: int = DEFAULT_PUBLIC_MAX_RESULTS,
    ) -> RetrievalResult:
        """Search the knowledge base and return ranked retrieval evidence.

        ``strategy`` is one of "lexical", "vector", "hybrid", or omitted to
        use the server's configured default strategy.
        """
        parsed_strategy = RetrievalStrategy(strategy) if strategy else None
        try:
            return knowledge_service.search(
                query, strategy=parsed_strategy, max_results=max_results
            )
        except _EXPECTED_ERRORS as error:
            raise ToolError(str(error)) from error

    @server.tool()
    async def get_document(document_id: str) -> DocumentLookupResult:
        """Look up one document by id. found=False means no such document exists."""
        try:
            document = knowledge_service.get_document(document_id)
        except PersistenceError as error:
            raise ToolError(str(error)) from error
        return DocumentLookupResult(found=document is not None, document=document)

    @server.tool()
    async def get_chunk(chunk_id: str) -> ChunkLookupResult:
        """Look up one chunk by id. found=False means no such chunk exists."""
        try:
            chunk = knowledge_service.get_chunk(chunk_id)
        except PersistenceError as error:
            raise ToolError(str(error)) from error
        return ChunkLookupResult(found=chunk is not None, chunk=chunk)

    return server


def run_stdio_server(config: AppConfig) -> None:
    """Compose a read-only KnowledgeService from config and serve it over MCP stdio.

    Never ingests, migrates, or rebuilds anything: the repository is
    opened read-only, and vector capability, when configured, is used only
    to embed incoming query text, never to write vectors.
    """
    repository = open_repository(
        config, vector_index_enabled=config.embeddings.enabled, read_only=True
    )
    try:
        vector_retriever = None
        if config.embeddings.enabled:
            embedding_provider = build_embedding_provider(config)
            vector_retriever = build_vector_retriever(embedding_provider, repository)
        knowledge_service = build_knowledge_service(
            config, repository, vector_retriever=vector_retriever
        )
        server = build_server(knowledge_service)
        server.run("stdio")
    finally:
        repository.close()
