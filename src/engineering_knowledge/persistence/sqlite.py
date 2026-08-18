"""SQLite implementation of the persistence Repository, LexicalIndex, and VectorIndex ports.

Owns one connection for its lifetime rather than opening one per call: a
``:memory:`` database only exists for as long as its connection is open, so
reopening per operation would silently discard everything between calls.
Foreign keys are enabled explicitly on that connection, since SQLite does
not enable FK enforcement by default.

All SQL and schema knowledge lives here; nothing above this module knows
what a table or a column is.

Vector capability is opt-in and lazy: ``sqlite_vec`` is only imported
inside ``_load_vector_extension``, called only when a repository is
constructed with ``vector_index_enabled=True``. A repository built without
it, or a database that has never had vectors built, keeps normalized state
and lexical search working without ever touching, importing, or requiring
the optional vector dependency.

``read_only=True`` opens the database file through SQLite's URI ``mode=ro``
semantics: a missing file is refused outright rather than created, no
schema migration ever runs (a version other than current is a hard stop),
and ``sync_source``/``rebuild_vector_index`` fail immediately at this
boundary rather than relying on SQLite eventually rejecting a write deep
inside a transaction. This is the mode knowledge-serving composition (the
MCP adapter) uses: it trusts an already-prepared database and never
ingests, migrates, or rebuilds anything.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Sequence
from datetime import datetime
from types import TracebackType
from typing import cast
from urllib.parse import quote

from engineering_knowledge.domain import (
    Chunk,
    Document,
    DocumentSource,
    SectionPath,
    SourceReference,
)
from engineering_knowledge.embeddings.base import EmbeddingProvider
from engineering_knowledge.embeddings.fingerprint import (
    build_embedding_text,
    derive_embedding_fingerprint,
)
from engineering_knowledge.persistence.base import (
    PersistenceError,
    ProcessedDocument,
    SourceSyncResult,
    UnsupportedSchemaVersionError,
    VectorIndexingSummary,
)
from engineering_knowledge.retrieval.errors import (
    IndexUnavailableError,
    InvalidQueryError,
    VectorIndexIncompatibleError,
    VectorIndexUnavailableError,
)
from engineering_knowledge.retrieval.lexical import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS,
    LexicalMatch,
    build_match_expression,
)
from engineering_knowledge.retrieval.vector import VectorMatch

SCHEMA_VERSION = 3

_NORMALIZED_SCHEMA_SQL = """
CREATE TABLE document_sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    display_name TEXT
);

CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES document_sources(source_id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    title TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    document_type TEXT,
    processing_fingerprint TEXT NOT NULL
);

CREATE UNIQUE INDEX documents_source_relative_path ON documents(source_id, relative_path);

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    section_path TEXT NOT NULL,
    section_occurrence INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    ordinal_in_section INTEGER NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    char_count INTEGER NOT NULL
);

CREATE INDEX chunks_document_id ON chunks(document_id);

CREATE TABLE vector_index_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    embedding_fingerprint TEXT NOT NULL,
    dimension INTEGER NOT NULL
);
"""

# A plain table, not a virtual one: no extension is needed to create, read,
# or migrate it, so it can live in the always-created normalized schema
# without breaking a base install's ability to open the database. It stays
# empty until vectors are actually built. The vec0 virtual table itself
# (which does need sqlite-vec) is created separately and lazily, in
# _ensure_vector_table, once a provider's dimension is known.
_VECTOR_STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vector_index_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    embedding_fingerprint TEXT NOT NULL,
    dimension INTEGER NOT NULL
);
"""

# unicode61 with '_' and '-' added as token characters: engineering
# identifiers like MAX_RETRY_COUNT or payments-service are what this
# system's queries actually look like, and splitting them into fragments
# on every underscore/hyphen would make exact-identifier search worse, not
# more flexible. No stemming: an identifier is not a natural-language word.
_FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE chunk_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    title,
    relative_path,
    section_path,
    text,
    tokenize = "unicode61 tokenchars '_-'"
);
"""

_DOCUMENT_COLUMNS = (
    "document_id, source_id, relative_path, title, content_hash, ingested_at, document_type"
)
_CHUNK_COLUMNS = (
    "chunk_id, document_id, section_path, section_occurrence, ordinal, "
    "ordinal_in_section, text, content_hash, char_count"
)
_CHUNK_FTS_COLUMNS = "chunk_id, document_id, title, relative_path, section_path, text"
_LEXICAL_SEARCH_SQL = """
SELECT
    chunks.chunk_id, chunks.document_id, chunks.section_path, chunks.section_occurrence,
    chunks.ordinal, chunks.ordinal_in_section, chunks.text, chunks.content_hash,
    chunks.char_count, documents.source_id, documents.relative_path, bm25(chunk_fts) AS score
FROM chunk_fts
JOIN chunks ON chunks.chunk_id = chunk_fts.chunk_id
JOIN documents ON documents.document_id = chunks.document_id
WHERE chunk_fts MATCH ?
ORDER BY score ASC, documents.relative_path ASC, chunks.ordinal ASC, chunks.chunk_id ASC
LIMIT ?
"""
_VECTOR_SEARCH_SQL = """
SELECT
    chunks.chunk_id, chunks.document_id, chunks.section_path, chunks.section_occurrence,
    chunks.ordinal, chunks.ordinal_in_section, chunks.text, chunks.content_hash,
    chunks.char_count, documents.source_id, documents.relative_path, vec_chunks.distance
FROM vec_chunks
JOIN chunks ON chunks.chunk_id = vec_chunks.chunk_id
JOIN documents ON documents.document_id = chunks.document_id
WHERE vec_chunks.embedding MATCH ? AND k = ?
ORDER BY vec_chunks.distance ASC, documents.relative_path ASC,
    chunks.ordinal ASC, chunks.chunk_id ASC
"""


class SqliteRepository:
    """A single-file (or ``:memory:``) SQLite Repository.

    Reads and writes are each wrapped in their own transaction; ``sqlite3``
    begins one implicitly on the first statement inside a ``with
    connection:`` block and commits on success or rolls back on any
    exception, which is what makes ``sync_source`` atomic without any
    manual BEGIN/COMMIT bookkeeping here.
    """

    def __init__(
        self,
        database_path: str,
        *,
        vector_index_enabled: bool = False,
        read_only: bool = False,
    ) -> None:
        self._read_only = read_only
        self._connection = (
            self._open_read_only(database_path) if read_only else sqlite3.connect(database_path)
        )
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
        except sqlite3.OperationalError as error:
            raise PersistenceError(f"failed to open database: {database_path}") from error
        self._vector_index_enabled = vector_index_enabled
        self._initialize_schema()
        if vector_index_enabled:
            self._load_vector_extension()

    @staticmethod
    def _open_read_only(database_path: str) -> sqlite3.Connection:
        # SQLite URI mode=ro refuses to create a missing file rather than
        # silently starting a fresh empty database, which is exactly the
        # "the database must already exist" guarantee read-only serving
        # depends on; a ":memory:" database has nothing to open read-only.
        if database_path == ":memory:":
            raise PersistenceError("read-only mode requires an on-disk database, not ':memory:'")
        uri = f"file:{quote(database_path)}?mode=ro"
        try:
            return sqlite3.connect(uri, uri=True)
        except sqlite3.OperationalError as error:
            raise PersistenceError(
                f"database does not exist or is not readable: {database_path}"
            ) from error

    def close(self) -> None:
        self._connection.close()

    def _require_writable(self, operation: str) -> None:
        # Failing here, before any statement runs, gives a clear typed
        # error instead of relying on SQLite eventually surfacing "attempt
        # to write a readonly database" from deep inside a transaction.
        if self._read_only:
            raise PersistenceError(f"{operation} is not permitted on a read-only repository")

    def __enter__(self) -> SqliteRepository:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _initialize_schema(self) -> None:
        if self._read_only:
            # Read-only serving trusts an already-prepared database: no
            # migration, no FTS/vector table creation, no schema writes.
            # A version other than current is a hard stop rather than an
            # upgrade attempt, since this connection cannot write one.
            version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    f"database schema version {version} is not supported by this build "
                    f"(expected {SCHEMA_VERSION}); read-only mode never migrates a schema"
                )
            return

        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            with self._connection:
                self._connection.executescript(_NORMALIZED_SCHEMA_SQL)
                self._create_fts_table()
                # PRAGMA does not accept bound parameters; SCHEMA_VERSION is
                # a fixed internal constant, never external input, so
                # formatting it directly into the statement is safe.
                self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        if version == 1:
            with self._connection:
                self._create_fts_table()
                self._backfill_fts_from_normalized_state()
                self._connection.execute(_VECTOR_STATE_TABLE_SQL)
                self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        if version == 2:
            with self._connection:
                self._connection.execute(_VECTOR_STATE_TABLE_SQL)
                self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        if version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(
                f"database schema version {version} is not supported by this build "
                f"(expected {SCHEMA_VERSION})"
            )

    def _create_fts_table(self) -> None:
        try:
            self._connection.execute(_FTS_SCHEMA_SQL)
        except sqlite3.OperationalError as error:
            if "no such module" in str(error).lower():
                raise PersistenceError(
                    "FTS5 is required for lexical retrieval but is not available "
                    "in this SQLite build"
                ) from error
            raise

    def _backfill_fts_from_normalized_state(self) -> None:
        # Migration is an index backfill only: it reads existing documents
        # and chunks exactly as ingestion already persisted them, and never
        # touches ingested_at, processing_fingerprint, or chunk content.
        rows = self._connection.execute(
            "SELECT chunks.chunk_id, chunks.document_id, chunks.section_path, chunks.text, "
            "documents.title, documents.relative_path "
            "FROM chunks JOIN documents ON documents.document_id = chunks.document_id"
        ).fetchall()
        self._connection.executemany(
            f"INSERT INTO chunk_fts ({_CHUNK_FTS_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    chunk_id,
                    document_id,
                    title,
                    relative_path,
                    _section_path_for_fts(
                        SectionPath(headings=tuple(json.loads(section_path_json)))
                    ),
                    text,
                )
                for chunk_id, document_id, section_path_json, text, title, relative_path in rows
            ],
        )

    def _load_vector_extension(self) -> None:
        try:
            import sqlite_vec
        except ImportError as error:
            raise VectorIndexUnavailableError(
                "vector index capability requires the optional 'sqlite-vec' package"
            ) from error

        # Loading is scoped as narrowly as possible: enabled immediately
        # before the one known extension load, disabled immediately after,
        # so this connection never exposes a general extension-loading
        # surface to anything else.
        try:
            self._connection.enable_load_extension(True)
        except AttributeError as error:
            raise VectorIndexUnavailableError(
                "this Python's sqlite3 build does not support loading extensions, "
                "which vector index capability requires"
            ) from error
        try:
            sqlite_vec.load(self._connection)
        except sqlite3.Error as error:
            raise VectorIndexUnavailableError("failed to load the sqlite-vec extension") from error
        finally:
            self._connection.enable_load_extension(False)

    def _ensure_vector_table(self, dimension: int) -> None:
        state = self._connection.execute(
            "SELECT dimension FROM vector_index_state WHERE id = 1"
        ).fetchone()
        table_exists = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vec_chunks'"
        ).fetchone()
        if table_exists is not None and state is not None and state[0] == dimension:
            return
        # A vec0 table's dimension is fixed at creation; a changed provider
        # dimension means the old table can no longer hold current vectors,
        # so it is dropped and recreated rather than altered in place.
        self._connection.execute("DROP TABLE IF EXISTS vec_chunks")
        self._connection.execute(
            "CREATE VIRTUAL TABLE vec_chunks USING vec0("
            "chunk_id TEXT PRIMARY KEY, document_id TEXT, "
            f"embedding FLOAT[{dimension}])"
        )

    def sync_source(
        self, source: DocumentSource, processed_documents: Sequence[ProcessedDocument]
    ) -> SourceSyncResult:
        self._require_writable("sync_source")
        incoming = {pd.document.document_id: pd for pd in processed_documents}
        created = updated = reprocessed = unchanged = vector_reindexed = 0

        # The embedding_fingerprint carried on incoming ProcessedDocuments
        # describes what these particular vectors were computed under. It
        # is never a request to change the database's global vector
        # configuration: only rebuild_vector_index may create the active
        # vector index or change its fingerprint/dimension. A source-scoped
        # sync only keeps an already-active, compatible index current.
        incoming_fingerprint = next(
            (pd.embedding_fingerprint for pd in processed_documents if pd.embedding_fingerprint),
            None,
        )

        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO document_sources (source_id, source_type, display_name)
                    VALUES (?, ?, ?)
                    ON CONFLICT(source_id) DO UPDATE SET
                        source_type = excluded.source_type,
                        display_name = excluded.display_name
                    """,
                    (source.source_id, source.source_type, source.display_name),
                )

                existing_rows = self._connection.execute(
                    "SELECT document_id, content_hash, processing_fingerprint "
                    "FROM documents WHERE source_id = ?",
                    (source.source_id,),
                ).fetchall()
                existing_state = {row[0]: (row[1], row[2]) for row in existing_rows}

                active_row = self._connection.execute(
                    "SELECT embedding_fingerprint FROM vector_index_state WHERE id = 1"
                ).fetchone()
                vector_index_active = active_row is not None

                # No active index: ordinary sync stays normalized/lexical-
                # only, even if an embedding_provider happened to be
                # configured. Activation is explicit, only through
                # rebuild_vector_index, never as a side effect of ingesting
                # whichever source happens to run first.
                write_vectors = False
                if vector_index_active:
                    if incoming_fingerprint is not None and incoming_fingerprint != active_row[0]:
                        # A source-scoped sync must never be allowed to
                        # silently repoint the global vector configuration:
                        # other sources' vectors would be left under the
                        # old fingerprint while this one moved to a new
                        # one, and vector_index_state would describe
                        # neither correctly.
                        raise VectorIndexIncompatibleError(
                            "ingestion embedding fingerprint does not match the active "
                            "vector index; run rebuild_vector_index to change the active "
                            "embedding configuration"
                        )
                    write_vectors = incoming_fingerprint is not None

                for document_id, processed in incoming.items():
                    previous = existing_state.get(document_id)
                    if previous is None:
                        self._require_vectors_for_write(vector_index_active, write_vectors)
                        self._write_document(processed, write_vectors=write_vectors)
                        created += 1
                        if write_vectors:
                            vector_reindexed += 1
                        continue

                    previous_content_hash, previous_fingerprint = previous
                    content_changed = previous_content_hash != processed.document.content_hash
                    fingerprint_changed = (
                        previous_fingerprint != processed.processing_fingerprint
                    )

                    if not content_changed and not fingerprint_changed:
                        # Neither the normalized content nor the processing
                        # behavior that would produce chunks from it has
                        # changed, so this row, including ingested_at, is
                        # left exactly as it is: a no-op re-ingestion must
                        # not look like a fresh one. Its existing vectors,
                        # if any, are left untouched too: reindexing an
                        # unchanged document is rebuild_vector_index's job,
                        # not ordinary sync's.
                        unchanged += 1
                        continue

                    self._require_vectors_for_write(vector_index_active, write_vectors)
                    self._write_document(processed, write_vectors=write_vectors)
                    if content_changed:
                        updated += 1
                    else:
                        reprocessed += 1
                    if write_vectors:
                        vector_reindexed += 1

                deleted_ids = set(existing_state) - set(incoming)
                for document_id in deleted_ids:
                    # FTS5 and vec0 virtual tables have no real foreign-key
                    # cascade, so a deleted document's index rows have to be
                    # removed explicitly rather than relying on ON DELETE
                    # CASCADE, which only applies to the normalized chunks
                    # table. No embedding model call is needed to delete.
                    self._connection.execute(
                        "DELETE FROM chunk_fts WHERE document_id = ?", (document_id,)
                    )
                    if vector_index_active:
                        self._connection.execute(
                            "DELETE FROM vec_chunks WHERE document_id = ?", (document_id,)
                        )
                    self._connection.execute(
                        "DELETE FROM documents WHERE document_id = ?", (document_id,)
                    )

                # sync_source never writes vector_index_state: only
                # rebuild_vector_index owns the active embedding
                # configuration.
        except sqlite3.Error as error:
            raise PersistenceError(f"failed to sync source {source.source_id!r}") from error

        return SourceSyncResult(
            created=created,
            updated=updated,
            reprocessed=reprocessed,
            unchanged=unchanged,
            deleted=len(deleted_ids),
            vector_reindexed=vector_reindexed,
        )

    @staticmethod
    def _require_vectors_for_write(vector_index_active: bool, write_vectors: bool) -> None:
        if vector_index_active and not write_vectors:
            raise VectorIndexIncompatibleError(
                "a vector index is active but no compatible embeddings were supplied "
                "for a document that needs writing; configure an embedding_provider "
                "matching the active embedding fingerprint, or run rebuild_vector_index "
                "first if the embedding configuration itself changed"
            )

    def _write_document(self, processed: ProcessedDocument, *, write_vectors: bool) -> None:
        document = processed.document
        self._connection.execute(
            """
            INSERT INTO documents (
                document_id, source_id, relative_path, title, content_hash,
                ingested_at, document_type, processing_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                relative_path = excluded.relative_path,
                title = excluded.title,
                content_hash = excluded.content_hash,
                ingested_at = excluded.ingested_at,
                document_type = excluded.document_type,
                processing_fingerprint = excluded.processing_fingerprint
            """,
            (
                document.document_id,
                document.source_id,
                document.relative_path,
                document.title,
                document.content_hash,
                document.ingested_at.isoformat(),
                document.document_type,
                processed.processing_fingerprint,
            ),
        )
        # The new chunk set may differ in membership from the old one, not
        # just in content (a smaller max_chunk_chars can add or remove
        # chunk boundaries), so replacing wholesale rather than trying to
        # diff chunk-by-chunk is what guarantees no orphaned chunk survives,
        # in both the normalized table and the FTS index.
        self._connection.execute(
            "DELETE FROM chunk_fts WHERE document_id = ?", (document.document_id,)
        )
        self._connection.execute(
            "DELETE FROM chunks WHERE document_id = ?", (document.document_id,)
        )
        self._connection.executemany(
            f"INSERT INTO chunks ({_CHUNK_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    json.dumps(list(chunk.section_path.headings)),
                    chunk.section_occurrence,
                    chunk.ordinal,
                    chunk.ordinal_in_section,
                    chunk.text,
                    chunk.content_hash,
                    chunk.char_count,
                )
                for chunk in processed.chunks
            ],
        )
        self._connection.executemany(
            f"INSERT INTO chunk_fts ({_CHUNK_FTS_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    document.title,
                    document.relative_path,
                    _section_path_for_fts(chunk.section_path),
                    chunk.text,
                )
                for chunk in processed.chunks
            ],
        )
        if write_vectors:
            self._write_vectors(processed)

    def _write_vectors(self, processed: ProcessedDocument) -> None:
        if not self._vector_index_enabled:
            raise VectorIndexUnavailableError(
                "ProcessedDocument carries vectors but this repository was not "
                "constructed with vector_index_enabled=True"
            )
        document_id = processed.document.document_id
        self._connection.execute(
            "DELETE FROM vec_chunks WHERE document_id = ?", (document_id,)
        )
        vectors_by_chunk_id = {
            record.chunk_id: record.embedding for record in processed.vectors or ()
        }
        self._connection.executemany(
            "INSERT INTO vec_chunks (chunk_id, document_id, embedding) VALUES (?, ?, ?)",
            [
                (
                    chunk.chunk_id,
                    document_id,
                    _serialize_vector(vectors_by_chunk_id[chunk.chunk_id]),
                )
                for chunk in processed.chunks
                if chunk.chunk_id in vectors_by_chunk_id
            ],
        )

    def get_document(self, document_id: str) -> Document | None:
        try:
            row = self._connection.execute(
                f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        except sqlite3.Error as error:
            raise PersistenceError(f"failed to read document {document_id!r}") from error
        return self._document_from_row(row) if row is not None else None

    def list_documents_for_source(self, source_id: str) -> tuple[Document, ...]:
        try:
            rows = self._connection.execute(
                f"SELECT {_DOCUMENT_COLUMNS} FROM documents "
                "WHERE source_id = ? ORDER BY relative_path",
                (source_id,),
            ).fetchall()
        except sqlite3.Error as error:
            raise PersistenceError(f"failed to list documents for {source_id!r}") from error
        return tuple(self._document_from_row(row) for row in rows)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        try:
            row = self._connection.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        except sqlite3.Error as error:
            raise PersistenceError(f"failed to read chunk {chunk_id!r}") from error
        return self._chunk_from_row(row) if row is not None else None

    def get_chunks(self, document_id: str) -> tuple[Chunk, ...]:
        try:
            rows = self._connection.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE document_id = ? ORDER BY ordinal",
                (document_id,),
            ).fetchall()
        except sqlite3.Error as error:
            raise PersistenceError(f"failed to read chunks for {document_id!r}") from error
        return tuple(self._chunk_from_row(row) for row in rows)

    def search(
        self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS
    ) -> tuple[LexicalMatch, ...]:
        """Rank chunks by BM25 against a user query, with provenance attached.

        ``query`` is ordinary text, never raw FTS syntax: it is converted
        into a safe MATCH expression by ``build_match_expression`` before it
        ever reaches SQLite, and that expression is itself passed as a bound
        parameter, not concatenated into the SQL statement.
        """
        if max_results <= 0:
            raise InvalidQueryError("max_results must be positive")
        if max_results > MAX_RESULTS:
            raise InvalidQueryError(f"max_results must not exceed {MAX_RESULTS}")

        match_expression = build_match_expression(query)

        try:
            rows = self._connection.execute(
                _LEXICAL_SEARCH_SQL, (match_expression, max_results)
            ).fetchall()
        except sqlite3.Error as error:
            raise IndexUnavailableError("lexical search failed") from error

        return tuple(
            self._lexical_match_from_row(row, rank) for rank, row in enumerate(rows, start=1)
        )

    def search_vector(
        self,
        query_vector: tuple[float, ...],
        *,
        embedding_fingerprint: str,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> tuple[VectorMatch, ...]:
        """Nearest-neighbor search against persisted chunk vectors, with provenance attached.

        ``embedding_fingerprint`` must match the active vector index's
        fingerprint: a query vector from a different provider/model is
        rejected rather than compared against vectors from a different
        embedding space, which would produce meaningless distances.
        """
        if not self._vector_index_enabled:
            raise VectorIndexUnavailableError(
                "vector index capability is not enabled for this repository"
            )
        if max_results <= 0:
            raise InvalidQueryError("max_results must be positive")
        if max_results > MAX_RESULTS:
            raise InvalidQueryError(f"max_results must not exceed {MAX_RESULTS}")

        state = self._connection.execute(
            "SELECT embedding_fingerprint, dimension FROM vector_index_state WHERE id = 1"
        ).fetchone()
        if state is None:
            raise VectorIndexUnavailableError("vector index has not been built yet")
        active_fingerprint, dimension = state
        if active_fingerprint != embedding_fingerprint:
            raise VectorIndexIncompatibleError(
                "query embedding fingerprint does not match the active vector index; "
                "rebuild the vector index"
            )
        if len(query_vector) != dimension:
            raise VectorIndexIncompatibleError(
                f"query vector dimension {len(query_vector)} does not match "
                f"index dimension {dimension}"
            )

        try:
            rows = self._connection.execute(
                _VECTOR_SEARCH_SQL, (_serialize_vector(query_vector), max_results)
            ).fetchall()
        except sqlite3.Error as error:
            raise VectorIndexUnavailableError("vector search failed") from error

        return tuple(
            self._vector_match_from_row(row, rank) for rank, row in enumerate(rows, start=1)
        )

    def rebuild_vector_index(self, embedding_provider: EmbeddingProvider) -> VectorIndexingSummary:
        """Rebuild the vector index from current normalized state, no source re-ingestion.

        Reads every persisted document/chunk directly, computes embedding
        text and vectors outside any write transaction, then atomically
        replaces the vector table and its metadata. Works whether this is
        the first time vectors are enabled, a model change, a dimension
        change, or recovery from stale/corrupted vector state; all of those
        are the same operation, since the vector index is fully derived
        from normalized state and never the other way around.
        """
        self._require_writable("rebuild_vector_index")
        if not self._vector_index_enabled:
            raise VectorIndexUnavailableError(
                "vector index capability is not enabled for this repository"
            )

        rows = self._connection.execute(
            "SELECT chunks.chunk_id, chunks.text, chunks.section_path, documents.title "
            "FROM chunks JOIN documents ON documents.document_id = chunks.document_id "
            "ORDER BY chunks.document_id, chunks.ordinal"
        ).fetchall()

        chunk_ids = [row[0] for row in rows]
        texts = [
            build_embedding_text(
                title=title,
                section_path=SectionPath(headings=tuple(json.loads(section_path_json))),
                chunk_text=text,
            )
            for _, text, section_path_json, title in rows
        ]
        vectors = embedding_provider.embed_documents(texts) if texts else ()

        embedding_fingerprint = derive_embedding_fingerprint(
            provider_type=embedding_provider.provider_type,
            model_id=embedding_provider.model_id,
            model_revision=embedding_provider.model_revision,
            dimension=embedding_provider.dimension,
        )

        try:
            with self._connection:
                self._ensure_vector_table(embedding_provider.dimension)
                self._connection.execute("DELETE FROM vec_chunks")
                document_id_by_chunk_id = self._connection.execute(
                    "SELECT chunk_id, document_id FROM chunks"
                ).fetchall()
                document_ids = dict(document_id_by_chunk_id)
                self._connection.executemany(
                    "INSERT INTO vec_chunks (chunk_id, document_id, embedding) VALUES (?, ?, ?)",
                    [
                        (chunk_id, document_ids[chunk_id], _serialize_vector(vector))
                        for chunk_id, vector in zip(chunk_ids, vectors, strict=True)
                    ],
                )
                self._connection.execute(
                    """
                    INSERT INTO vector_index_state (id, embedding_fingerprint, dimension)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        embedding_fingerprint = excluded.embedding_fingerprint,
                        dimension = excluded.dimension
                    """,
                    (embedding_fingerprint, embedding_provider.dimension),
                )
        except sqlite3.Error as error:
            raise VectorIndexUnavailableError("failed to rebuild vector index") from error

        return VectorIndexingSummary(
            embedding_fingerprint=embedding_fingerprint, reindexed=len(chunk_ids)
        )

    @staticmethod
    def _document_from_row(row: Sequence[object]) -> Document:
        # sqlite3's stubs return Any for a fetched row; cast() here is
        # purely a type hint for mypy, trusting the schema we wrote, not a
        # runtime validation of externally supplied data.
        document_id, source_id, relative_path, title, content_hash, ingested_at, doc_type = (
            cast("tuple[str, str, str, str, str, str, str | None]", tuple(row))
        )
        return Document(
            document_id=document_id,
            source_id=source_id,
            relative_path=relative_path,
            title=title,
            content_hash=content_hash,
            ingested_at=datetime.fromisoformat(ingested_at),
            document_type=doc_type,
        )

    @staticmethod
    def _chunk_from_row(row: Sequence[object]) -> Chunk:
        (
            chunk_id,
            document_id,
            section_path_json,
            section_occurrence,
            ordinal,
            ordinal_in_section,
            text,
            content_hash,
            char_count,
        ) = cast("tuple[str, str, str, int, int, int, str, str, int]", tuple(row))
        return Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            section_path=SectionPath(headings=tuple(json.loads(section_path_json))),
            section_occurrence=section_occurrence,
            ordinal=ordinal,
            ordinal_in_section=ordinal_in_section,
            text=text,
            content_hash=content_hash,
            char_count=char_count,
        )

    @staticmethod
    def _lexical_match_from_row(row: Sequence[object], rank: int) -> LexicalMatch:
        (
            chunk_id,
            document_id,
            section_path_json,
            section_occurrence,
            ordinal,
            ordinal_in_section,
            text,
            content_hash,
            char_count,
            source_id,
            relative_path,
            score,
        ) = cast(
            "tuple[str, str, str, int, int, int, str, str, int, str, str, float]", tuple(row)
        )
        section_path = SectionPath(headings=tuple(json.loads(section_path_json)))
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            section_path=section_path,
            section_occurrence=section_occurrence,
            ordinal=ordinal,
            ordinal_in_section=ordinal_in_section,
            text=text,
            content_hash=content_hash,
            char_count=char_count,
        )
        source_reference = SourceReference(
            source_id=source_id,
            document_id=document_id,
            chunk_id=chunk_id,
            relative_path=relative_path,
            section_path=section_path,
            section_occurrence=section_occurrence,
            content_hash=content_hash,
        )
        return LexicalMatch(
            chunk=chunk, source_reference=source_reference, rank=rank, bm25_score=score
        )

    @staticmethod
    def _vector_match_from_row(row: Sequence[object], rank: int) -> VectorMatch:
        (
            chunk_id,
            document_id,
            section_path_json,
            section_occurrence,
            ordinal,
            ordinal_in_section,
            text,
            content_hash,
            char_count,
            source_id,
            relative_path,
            distance,
        ) = cast(
            "tuple[str, str, str, int, int, int, str, str, int, str, str, float]", tuple(row)
        )
        section_path = SectionPath(headings=tuple(json.loads(section_path_json)))
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            section_path=section_path,
            section_occurrence=section_occurrence,
            ordinal=ordinal,
            ordinal_in_section=ordinal_in_section,
            text=text,
            content_hash=content_hash,
            char_count=char_count,
        )
        source_reference = SourceReference(
            source_id=source_id,
            document_id=document_id,
            chunk_id=chunk_id,
            relative_path=relative_path,
            section_path=section_path,
            section_occurrence=section_occurrence,
            content_hash=content_hash,
        )
        return VectorMatch(
            chunk=chunk, source_reference=source_reference, rank=rank, distance=distance
        )


def _section_path_for_fts(section_path: SectionPath) -> str:
    """Space-joined heading text, empty for the root section.

    Deliberately not the JSON representation the chunks table stores: this
    is only meant to let heading terms participate in lexical matching, not
    to preserve structure, which normalized persistence already owns.
    """
    return " ".join(section_path.headings)


def _serialize_vector(vector: Sequence[float]) -> bytes:
    """Pack a vector into sqlite-vec's expected little-endian float32 layout."""
    return struct.pack(f"{len(vector)}f", *vector)
