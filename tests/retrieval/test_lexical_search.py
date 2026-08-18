import json
import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engineering_knowledge.domain import SectionPath
from engineering_knowledge.domain.identity import derive_chunk_id, derive_document_id, hash_content
from engineering_knowledge.ingestion import IngestionService
from engineering_knowledge.persistence import SqliteRepository
from engineering_knowledge.retrieval import MAX_RESULTS, InvalidQueryError
from engineering_knowledge.sources import LocalFilesystemSourceAdapter

_V1_SCHEMA_SQL = """
CREATE TABLE document_sources (
    source_id TEXT PRIMARY KEY, source_type TEXT NOT NULL, display_name TEXT
);
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES document_sources(source_id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL, title TEXT NOT NULL, content_hash TEXT NOT NULL,
    ingested_at TEXT NOT NULL, document_type TEXT, processing_fingerprint TEXT NOT NULL
);
CREATE UNIQUE INDEX documents_source_relative_path ON documents(source_id, relative_path);
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    section_path TEXT NOT NULL, section_occurrence INTEGER NOT NULL, ordinal INTEGER NOT NULL,
    ordinal_in_section INTEGER NOT NULL, text TEXT NOT NULL, content_hash TEXT NOT NULL,
    char_count INTEGER NOT NULL
);
CREATE INDEX chunks_document_id ON chunks(document_id);
"""


def _make_clock(*timestamps: datetime) -> Callable[[], datetime]:
    values: Iterator[datetime] = iter(timestamps)
    return lambda: next(values)


def test_search_finds_engineering_identifiers_with_structured_provenance(
    tmp_path: Path,
) -> None:
    (tmp_path / "troubleshooting.md").write_text(
        "# Aegis Troubleshooting\n\n"
        "## PaymentGatewayTimeoutError\n\n"
        "Check MAX_RETRY_COUNT on the payments-service deployment.\n"
    )
    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000, clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC))
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path))

    matches = repo.search("PaymentGatewayTimeoutError")
    assert len(matches) == 1
    match = matches[0]
    assert match.rank == 1
    assert "PaymentGatewayTimeoutError" in match.chunk.text
    assert match.source_reference.source_id == "docs"
    assert match.source_reference.relative_path == "troubleshooting.md"
    assert match.source_reference.section_path.headings == (
        "Aegis Troubleshooting",
        "PaymentGatewayTimeoutError",
    )
    assert match.source_reference.chunk_id == match.chunk.chunk_id
    assert match.source_reference.content_hash == match.chunk.content_hash

    assert len(repo.search("MAX_RETRY_COUNT")) == 1
    assert len(repo.search("payments-service")) == 1

    repo.close()


@pytest.mark.parametrize(
    "query",
    [
        '"quoted" text',
        "()[]{}:^*",
        "payments-service AND rollback OR NOT deploy",
    ],
)
def test_search_handles_punctuation_and_operator_lookalikes_without_crashing(
    tmp_path: Path, query: str,
) -> None:
    (tmp_path / "notes.md").write_text("# Notes\n\npayments-service rollback deploy content.\n")
    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000, clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC))
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path))

    matches = repo.search(query)
    assert isinstance(matches, tuple)

    repo.close()


def test_search_no_match_returns_empty_tuple(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Notes\n\nsome ordinary content.\n")
    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000, clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC))
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path))

    assert repo.search("nonexistent_term_zzz") == ()

    repo.close()


@pytest.mark.parametrize(
    ("query", "max_results"),
    [
        ("", 10),
        ("   ", 10),
        ("valid query", 0),
        ("valid query", MAX_RESULTS + 1),
    ],
)
def test_search_rejects_invalid_query_and_bounds(query: str, max_results: int) -> None:
    repo = SqliteRepository(":memory:")
    with pytest.raises(InvalidQueryError):
        repo.search(query, max_results=max_results)
    repo.close()


def test_search_index_stays_consistent_through_update_reprocess_and_delete(
    tmp_path: Path,
) -> None:
    doc_path = tmp_path / "runbook.md"
    doc_path.write_text("# Payments\n\nOriginalMarkerTerm content here.\n")

    repo = SqliteRepository(":memory:")
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path)
    wide_service = IngestionService(
        repo, max_chunk_chars=1000,
        clock=_make_clock(
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            datetime(2026, 1, 4, tzinfo=UTC),
        ),
    )

    wide_service.ingest_source(adapter)
    assert len(repo.search("OriginalMarkerTerm")) == 1

    doc_path.write_text("# Payments\n\nReplacedMarkerTerm content here.\n")
    wide_service.ingest_source(adapter)
    assert repo.search("OriginalMarkerTerm") == ()
    assert len(repo.search("ReplacedMarkerTerm")) == 1

    # A different bound is enough to force REPROCESSED via a changed
    # processing_fingerprint; it is chosen well above the marker term's own
    # line length so the term is never split mid-word by the new bound.
    narrow_service = IngestionService(
        repo, max_chunk_chars=30, clock=_make_clock(datetime(2026, 1, 3, tzinfo=UTC))
    )
    narrow_service.ingest_source(adapter)
    assert len(repo.search("ReplacedMarkerTerm")) == 1

    doc_path.unlink()
    wide_service.ingest_source(adapter)
    assert repo.search("ReplacedMarkerTerm") == ()

    repo.close()


def test_search_ranking_is_deterministic_with_tie_break(tmp_path: Path) -> None:
    (tmp_path / "b_doc.md").write_text("# B\n\nsharedterm appears once here.\n")
    (tmp_path / "a_doc.md").write_text("# A\n\nsharedterm appears once here.\n")
    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000, clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC))
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path))

    first = repo.search("sharedterm")
    second = repo.search("sharedterm")

    assert first == second
    assert len(first) == 2
    assert first[0].bm25_score == first[1].bm25_score
    assert [m.source_reference.relative_path for m in first] == ["a_doc.md", "b_doc.md"]
    assert [m.rank for m in first] == [1, 2]

    repo.close()


def test_migration_v1_to_v3_backfills_fts_and_adds_inactive_vector_state(
    tmp_path: Path,
) -> None:
    # A schema-v1 database, built directly rather than through the current
    # repository, since migration behavior is defined against the historical
    # shape it upgrades from, not against whatever the current code creates.
    db_path = str(tmp_path / "v1.db")
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_V1_SCHEMA_SQL)

    document_id = derive_document_id("docs", "runbook.md")
    connection.execute(
        "INSERT INTO document_sources VALUES ('docs', 'local_filesystem', NULL)"
    )
    connection.execute(
        "INSERT INTO documents VALUES (?, 'docs', 'runbook.md', 'runbook.md', ?, "
        "'2026-01-01T00:00:00+00:00', NULL, 'fp1')",
        (document_id, hash_content("doc content")),
    )
    section = SectionPath(headings=("Payments",))
    chunk_id = derive_chunk_id(document_id, section.as_identity_string(), 0, 0)
    chunk_text = "LegacyBackfillTerm text"
    connection.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, 0, 0, 0, ?, ?, ?)",
        (
            chunk_id, document_id, json.dumps(["Payments"]),
            chunk_text, hash_content(chunk_text), len(chunk_text),
        ),
    )
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    repo = SqliteRepository(db_path)

    version = repo._connection.execute("PRAGMA user_version").fetchone()[0]
    assert version == 3

    matches = repo.search("LegacyBackfillTerm")
    assert len(matches) == 1
    assert matches[0].chunk.chunk_id == chunk_id
    assert matches[0].source_reference.relative_path == "runbook.md"

    document = repo.get_document(document_id)
    assert document is not None
    assert document.ingested_at == datetime(2026, 1, 1, tzinfo=UTC)

    # vector_index_state exists structurally (a plain table, no sqlite-vec
    # needed) but carries no active configuration: migration never
    # activates vector capability on its own, only rebuild_vector_index does.
    active_vector_state = repo._connection.execute(
        "SELECT * FROM vector_index_state WHERE id = 1"
    ).fetchone()
    assert active_vector_state is None

    repo.close()
