import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engineering_knowledge.ingestion import IngestionService
from engineering_knowledge.persistence import PersistenceError, SqliteRepository
from engineering_knowledge.persistence.base import UnsupportedSchemaVersionError
from engineering_knowledge.sources import LocalFilesystemSourceAdapter


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_only_repository_serves_reads_and_rejects_writes_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "runbook.md").write_text("# Payments\n\nRollback steps for Aegis payments.\n")
    db_path = tmp_path / "ek.db"

    writable = SqliteRepository(str(db_path))
    service = IngestionService(
        writable, max_chunk_chars=1000, clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=root))
    document = writable.list_documents_for_source("docs")[0]
    chunk = writable.get_chunks(document.document_id)[0]
    writable.close()

    before_hash = _file_hash(db_path)

    read_only = SqliteRepository(str(db_path), read_only=True)
    assert read_only.get_document(document.document_id) == document
    assert read_only.get_chunk(chunk.chunk_id) == chunk
    assert read_only.search("rollback")

    with pytest.raises(PersistenceError):
        read_only.sync_source(None, [])  # type: ignore[arg-type]
    with pytest.raises(PersistenceError):
        read_only.rebuild_vector_index(None)  # type: ignore[arg-type]

    read_only.close()

    assert _file_hash(db_path) == before_hash

    # Read-only serving trusts an already-prepared database: a missing file
    # is refused rather than created, and a schema version other than
    # current is a hard stop rather than a migration attempt.
    missing_path = tmp_path / "missing.db"
    with pytest.raises(PersistenceError):
        SqliteRepository(str(missing_path), read_only=True)

    stale_path = tmp_path / "stale.db"
    connection = sqlite3.connect(str(stale_path))
    connection.execute("CREATE TABLE placeholder (id INTEGER)")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    with pytest.raises(UnsupportedSchemaVersionError):
        SqliteRepository(str(stale_path), read_only=True)
