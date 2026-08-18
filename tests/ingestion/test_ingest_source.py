from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engineering_knowledge.domain.identity import hash_content
from engineering_knowledge.ingestion import IngestionResult, IngestionService
from engineering_knowledge.persistence import SqliteRepository
from engineering_knowledge.sources import LocalFilesystemSourceAdapter
from engineering_knowledge.sources.base import SourceReadError


def _make_clock(*timestamps: datetime) -> Callable[[], datetime]:
    values: Iterator[datetime] = iter(timestamps)
    return lambda: next(values)


def test_ingest_source_persists_source_documents_and_chunks(tmp_path: Path) -> None:
    (tmp_path / "runbook.md").write_text(
        "# Payments\n\nRollback steps.\n\n## Details\n\nMore detail text here.\n"
    )
    (tmp_path / "notes.txt").write_text("plain notes\nsecond line\n")

    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000, clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC))
    )
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path, display_name="Docs")

    result = service.ingest_source(adapter)

    assert result == IngestionResult(
        source_id="docs", discovered=2, created=2, updated=0,
        reprocessed=0, unchanged=0, deleted=0,
    )

    documents = repo.list_documents_for_source("docs")
    assert [d.relative_path for d in documents] == ["notes.txt", "runbook.md"]

    runbook = next(d for d in documents if d.relative_path == "runbook.md")
    chunks = repo.get_chunks(runbook.document_id)
    assert [c.section_path.headings for c in chunks] == [("Payments",), ("Payments", "Details")]
    assert [c.ordinal for c in chunks] == [0, 1]
    assert all(c.section_occurrence == 0 for c in chunks)
    assert all(c.content_hash == hash_content(c.text) for c in chunks)

    notes = next(d for d in documents if d.relative_path == "notes.txt")
    notes_chunks = repo.get_chunks(notes.document_id)
    assert notes_chunks[0].section_path.headings == ()

    repo.close()


def test_ingest_source_idempotent_reingestion_is_a_noop(tmp_path: Path) -> None:
    (tmp_path / "runbook.md").write_text("# Payments\n\nRollback steps.\n")
    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000,
        clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)),
    )
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path)

    service.ingest_source(adapter)
    document_id = repo.list_documents_for_source("docs")[0].document_id
    chunks_before = repo.get_chunks(document_id)

    second = service.ingest_source(adapter)

    assert second.unchanged == 1
    assert second.created == 0
    assert second.updated == 0
    assert second.reprocessed == 0

    document = repo.get_document(document_id)
    assert document is not None
    assert document.ingested_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert repo.get_chunks(document_id) == chunks_before

    repo.close()


def test_ingest_source_content_change_updates_document_and_replaces_chunks(
    tmp_path: Path,
) -> None:
    doc_path = tmp_path / "runbook.md"
    doc_path.write_text("# Payments\n\nOriginal rollback steps.\n")
    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000,
        clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)),
    )
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path)

    service.ingest_source(adapter)
    document_id = repo.list_documents_for_source("docs")[0].document_id
    original = repo.get_document(document_id)
    assert original is not None

    doc_path.write_text(
        "# Payments\n\nOriginal rollback steps.\n\n## Verification\n\nCheck logs.\n"
    )
    result = service.ingest_source(adapter)

    assert result.updated == 1
    assert result.unchanged == 0

    document = repo.get_document(document_id)
    assert document is not None
    assert document.document_id == document_id
    assert document.content_hash != original.content_hash
    assert document.ingested_at == datetime(2026, 1, 2, tzinfo=UTC)

    chunks = repo.get_chunks(document_id)
    assert [c.section_path.headings for c in chunks] == [
        ("Payments",),
        ("Payments", "Verification"),
    ]

    repo.close()


def test_ingest_source_processing_fingerprint_change_reprocesses_unchanged_content(
    tmp_path: Path,
) -> None:
    long_paragraph = "word " * 40
    (tmp_path / "runbook.md").write_text(f"# Payments\n\n{long_paragraph}\n")
    repo = SqliteRepository(":memory:")
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path)

    wide_service = IngestionService(
        repo, max_chunk_chars=1000, clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC))
    )
    wide_service.ingest_source(adapter)
    document_id = repo.list_documents_for_source("docs")[0].document_id
    wide_chunks = repo.get_chunks(document_id)
    original = repo.get_document(document_id)
    assert original is not None
    assert len(wide_chunks) == 1

    narrow_service = IngestionService(
        repo, max_chunk_chars=40, clock=_make_clock(datetime(2026, 1, 2, tzinfo=UTC))
    )
    result = narrow_service.ingest_source(adapter)

    assert result.reprocessed == 1
    assert result.unchanged == 0
    assert result.updated == 0

    document = repo.get_document(document_id)
    assert document is not None
    # the normalized source content never changed, only the bound used to
    # chunk it did, so content_hash must stay identical while chunks change
    assert document.content_hash == original.content_hash
    assert document.ingested_at == datetime(2026, 1, 2, tzinfo=UTC)

    narrow_chunks = repo.get_chunks(document_id)
    assert len(narrow_chunks) != len(wide_chunks)
    assert {c.content_hash for c in narrow_chunks} != {c.content_hash for c in wide_chunks}

    # chunk_id depends only on (document_id, section_path, section_occurrence,
    # ordinal_in_section), never on text or content_hash, so a chunk that
    # keeps the same logical position after reprocessing keeps the same
    # chunk_id even though a smaller bound changed its text and content_hash.
    # A processing-config change does not automatically invalidate every
    # chunk_id, only the ones whose position actually stopped existing.
    wide_by_position = {
        (c.section_path.headings, c.section_occurrence, c.ordinal_in_section): c.chunk_id
        for c in wide_chunks
    }
    narrow_by_position = {
        (c.section_path.headings, c.section_occurrence, c.ordinal_in_section): c.chunk_id
        for c in narrow_chunks
    }
    shared_positions = wide_by_position.keys() & narrow_by_position.keys()
    assert shared_positions
    assert all(wide_by_position[pos] == narrow_by_position[pos] for pos in shared_positions)

    repo.close()


def test_ingest_source_deleted_file_removes_document_and_chunks(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("# Keep\nstays.\n")
    remove_path = tmp_path / "remove.md"
    remove_path.write_text("# Remove\ngoes away.\n")

    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000,
        clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)),
    )
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path)

    service.ingest_source(adapter)
    removed_document_id = next(
        d.document_id
        for d in repo.list_documents_for_source("docs")
        if d.relative_path == "remove.md"
    )

    remove_path.unlink()
    result = service.ingest_source(adapter)

    assert result.deleted == 1
    assert result.discovered == 1
    assert repo.get_document(removed_document_id) is None
    assert repo.get_chunks(removed_document_id) == ()
    assert [d.relative_path for d in repo.list_documents_for_source("docs")] == ["keep.md"]

    repo.close()


def test_ingest_source_failed_discovery_leaves_existing_snapshot_unchanged(
    tmp_path: Path,
) -> None:
    (tmp_path / "runbook.md").write_text("# Payments\nRollback steps.\n")
    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000,
        clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)),
    )
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path)

    service.ingest_source(adapter)
    document_id = repo.list_documents_for_source("docs")[0].document_id
    chunks_before = repo.get_chunks(document_id)

    (tmp_path / "broken.md").write_bytes(b"\xff\xfe not valid utf-8")

    with pytest.raises(SourceReadError):
        service.ingest_source(adapter)

    document = repo.get_document(document_id)
    assert document is not None
    assert document.ingested_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert repo.get_chunks(document_id) == chunks_before
    assert [d.relative_path for d in repo.list_documents_for_source("docs")] == ["runbook.md"]

    repo.close()


def test_ingest_source_same_source_id_across_different_roots_shares_document_identity(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    root_a = tmp_path_factory.mktemp("checkout_a")
    root_b = tmp_path_factory.mktemp("checkout_b")
    content = "# Payments\nRollback steps.\n"
    (root_a / "runbook.md").write_text(content)
    (root_b / "runbook.md").write_text(content)

    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000,
        clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)),
    )

    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=root_a))
    documents_after_a = repo.list_documents_for_source("docs")
    assert len(documents_after_a) == 1

    result_b = service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=root_b))

    assert result_b.unchanged == 1
    assert result_b.created == 0
    documents_after_b = repo.list_documents_for_source("docs")
    assert len(documents_after_b) == 1
    assert documents_after_b[0].document_id == documents_after_a[0].document_id
    assert documents_after_b[0].relative_path == "runbook.md"

    repo.close()
