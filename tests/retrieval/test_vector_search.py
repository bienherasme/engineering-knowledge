from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

# vector_index_enabled=True loads the optional sqlite-vec extension; skip
# this whole module cleanly in a base-only environment rather than failing,
# since lexical/normalized functionality never requires this dependency.
pytest.importorskip("sqlite_vec")

from engineering_knowledge.embeddings import FakeEmbeddingProvider
from engineering_knowledge.ingestion import IngestionService
from engineering_knowledge.persistence import SqliteRepository
from engineering_knowledge.retrieval import MAX_RESULTS, InvalidQueryError, VectorRetriever
from engineering_knowledge.retrieval.errors import VectorIndexIncompatibleError
from engineering_knowledge.sources import LocalFilesystemSourceAdapter


def _make_clock(*timestamps: datetime) -> Callable[[], datetime]:
    values: Iterator[datetime] = iter(timestamps)
    return lambda: next(values)


def test_vector_search_persists_survives_reopen_with_structured_provenance(
    tmp_path: Path,
) -> None:
    db_path = str(tmp_path / "db.sqlite3")
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "runbook.md").write_text("# Payments\n\nRollback steps for Aegis.\n")

    provider = FakeEmbeddingProvider(dimension=4)
    repo = SqliteRepository(db_path, vector_index_enabled=True)
    service = IngestionService(
        repo, max_chunk_chars=1000, embedding_provider=provider,
        clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC)),
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=root))
    # activation is explicit: ordinary sync never establishes the active
    # vector index on its own, so rebuild_vector_index is required once
    # before vectors exist to search.
    repo.rebuild_vector_index(provider)
    repo.close()

    reopened = SqliteRepository(db_path, vector_index_enabled=True)
    retriever = VectorRetriever(provider, reopened)
    matches = retriever.search("rollback")

    assert len(matches) == 1
    match = matches[0]
    assert match.rank == 1
    assert match.chunk.text.startswith("# Payments")
    assert match.source_reference.source_id == "docs"
    assert match.source_reference.relative_path == "runbook.md"
    assert match.source_reference.chunk_id == match.chunk.chunk_id
    assert match.source_reference.content_hash == match.chunk.content_hash

    reopened.close()


def test_vector_index_stays_consistent_through_update_and_delete(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    doc_path = root / "runbook.md"
    doc_path.write_text("# Payments\n\noriginal content here.\n")

    provider = FakeEmbeddingProvider(dimension=4)
    repo = SqliteRepository(":memory:", vector_index_enabled=True)
    service = IngestionService(
        repo, max_chunk_chars=1000, embedding_provider=provider,
        clock=_make_clock(
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            datetime(2026, 1, 3, tzinfo=UTC),
        ),
    )
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=root)
    service.ingest_source(adapter)
    repo.rebuild_vector_index(provider)

    retriever = VectorRetriever(provider, repo)
    first_matches = retriever.search("query text", max_results=50)
    assert len(first_matches) == 1
    original_chunk_id = first_matches[0].chunk.chunk_id

    doc_path.write_text("# Payments\n\nreplaced content here.\n")
    service.ingest_source(adapter)
    second_matches = retriever.search("query text", max_results=50)
    assert len(second_matches) == 1
    assert second_matches[0].chunk.chunk_id == original_chunk_id
    assert second_matches[0].chunk.content_hash != first_matches[0].chunk.content_hash

    doc_path.unlink()
    service.ingest_source(adapter)
    assert retriever.search("query text", max_results=50) == ()

    repo.close()


def test_rebuild_vector_index_changes_embedding_state_without_touching_normalized_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "runbook.md").write_text("# Payments\n\nRollback steps for Aegis.\n")

    repo = SqliteRepository(":memory:", vector_index_enabled=True)
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=root)

    provider_a = FakeEmbeddingProvider(dimension=4)
    service = IngestionService(
        repo, max_chunk_chars=1000, embedding_provider=provider_a,
        clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC)),
    )
    service.ingest_source(adapter)
    repo.rebuild_vector_index(provider_a)

    document_id = repo.list_documents_for_source("docs")[0].document_id
    before_document = repo.get_document(document_id)
    before_chunks = repo.get_chunks(document_id)
    assert before_document is not None

    # Changing the active embedding configuration is exclusively
    # rebuild_vector_index's job, never a side effect of ingesting a
    # source: it is the only operation that can rebuild the entire derived
    # index coherently, since it reads every persisted document/chunk, not
    # just the ones a single source sync happens to touch.
    provider_b = FakeEmbeddingProvider(dimension=6)
    summary = repo.rebuild_vector_index(provider_b)
    assert summary.reindexed == 1

    after_document = repo.get_document(document_id)
    after_chunks = repo.get_chunks(document_id)
    assert after_document is not None

    assert after_document.document_id == before_document.document_id
    assert after_document.content_hash == before_document.content_hash
    assert after_document.ingested_at == before_document.ingested_at
    assert [c.chunk_id for c in after_chunks] == [c.chunk_id for c in before_chunks]
    assert [c.content_hash for c in after_chunks] == [c.content_hash for c in before_chunks]

    # vectors are genuinely incompatible now: querying with the old
    # provider against the currently active index is rejected outright,
    # not silently compared across embedding spaces
    retriever_a = VectorRetriever(provider_a, repo)
    with pytest.raises(VectorIndexIncompatibleError):
        retriever_a.search("rollback")

    retriever_b = VectorRetriever(provider_b, repo)
    assert len(retriever_b.search("rollback")) == 1

    repo.close()


def test_multi_source_vector_state_stays_globally_coherent(tmp_path: Path) -> None:
    root_a = tmp_path / "source_a"
    root_b = tmp_path / "source_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "doc.md").write_text("# A\ncontent for source A.\n")
    (root_b / "doc.md").write_text("# B\ncontent for source B.\n")

    repo = SqliteRepository(":memory:", vector_index_enabled=True)
    provider_a = FakeEmbeddingProvider(dimension=4)
    adapter_a = LocalFilesystemSourceAdapter(source_id="source-a", root=root_a)
    adapter_b = LocalFilesystemSourceAdapter(source_id="source-b", root=root_b)

    service_a = IngestionService(
        repo, max_chunk_chars=1000, embedding_provider=provider_a,
        clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)),
    )
    service_a.ingest_source(adapter_a)
    service_a.ingest_source(adapter_b)
    repo.rebuild_vector_index(provider_a)

    both_under_a = VectorRetriever(provider_a, repo).search("content", max_results=10)
    assert {m.source_reference.source_id for m in both_under_a} == {"source-a", "source-b"}

    # attempt source-A-only ingestion under an incompatible provider
    provider_b = FakeEmbeddingProvider(dimension=6)
    (root_a / "doc.md").write_text("# A\nchanged content for source A.\n")
    service_b = IngestionService(
        repo, max_chunk_chars=1000, embedding_provider=provider_b,
        clock=_make_clock(datetime(2026, 1, 3, tzinfo=UTC)),
    )
    document_a_before = repo.list_documents_for_source("source-a")[0]

    with pytest.raises(VectorIndexIncompatibleError):
        service_b.ingest_source(adapter_a)

    # a source-scoped sync must never be able to repoint the global vector
    # configuration or leave sources split across two embedding spaces
    document_a_after = repo.list_documents_for_source("source-a")[0]
    assert document_a_after.content_hash == document_a_before.content_hash
    still_under_a = VectorRetriever(provider_a, repo).search("content", max_results=10)
    assert {m.source_reference.source_id for m in still_under_a} == {"source-a", "source-b"}

    # the only correct way to move the active configuration to provider B
    # is a global rebuild, which covers every persisted source at once
    summary = repo.rebuild_vector_index(provider_b)
    assert summary.reindexed == 2
    both_under_b = VectorRetriever(provider_b, repo).search("content", max_results=10)
    assert {m.source_reference.source_id for m in both_under_b} == {"source-a", "source-b"}

    repo.close()


def test_active_index_without_provider_fails_atomically(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    doc_path = root / "runbook.md"
    doc_path.write_text("# Payments\n\noriginal content.\n")

    provider = FakeEmbeddingProvider(dimension=4)
    repo = SqliteRepository(":memory:", vector_index_enabled=True)
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=root)
    service_with_provider = IngestionService(
        repo, max_chunk_chars=1000, embedding_provider=provider,
        clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC)),
    )
    service_with_provider.ingest_source(adapter)
    repo.rebuild_vector_index(provider)

    document_id = repo.list_documents_for_source("docs")[0].document_id
    before_document = repo.get_document(document_id)
    before_chunks = repo.get_chunks(document_id)
    before_lexical = repo.search("original")
    before_vector = VectorRetriever(provider, repo).search("content", max_results=10)
    assert before_document is not None

    doc_path.write_text("# Payments\n\nchanged content that would update chunks.\n")
    service_without_provider = IngestionService(
        repo, max_chunk_chars=1000, clock=_make_clock(datetime(2026, 1, 2, tzinfo=UTC))
    )

    with pytest.raises(VectorIndexIncompatibleError):
        service_without_provider.ingest_source(adapter)

    # the whole sync must roll back, not just skip the vector write:
    # normalized state, FTS, and vectors all stay the previous snapshot
    after_document = repo.get_document(document_id)
    after_chunks = repo.get_chunks(document_id)
    after_lexical = repo.search("original")
    after_vector = VectorRetriever(provider, repo).search("content", max_results=10)

    assert after_document is not None
    assert after_document.content_hash == before_document.content_hash
    assert after_document.ingested_at == before_document.ingested_at
    assert [c.chunk_id for c in after_chunks] == [c.chunk_id for c in before_chunks]
    assert [c.content_hash for c in after_chunks] == [c.content_hash for c in before_chunks]
    assert after_lexical == before_lexical
    assert after_vector == before_vector

    repo.close()


def test_rebuild_vector_index_from_existing_database_without_reingestion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "runbook.md").write_text("# Payments\n\nRollback steps.\n")
    db_path = str(tmp_path / "db.sqlite3")

    lexical_repo = SqliteRepository(db_path)
    service = IngestionService(
        lexical_repo, max_chunk_chars=1000, clock=_make_clock(datetime(2026, 1, 1, tzinfo=UTC))
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=root))
    lexical_repo.close()

    vector_repo = SqliteRepository(db_path, vector_index_enabled=True)
    provider = FakeEmbeddingProvider(dimension=4)
    summary = vector_repo.rebuild_vector_index(provider)
    assert summary.reindexed == 1

    matches = VectorRetriever(provider, vector_repo).search("rollback")
    assert len(matches) == 1
    assert matches[0].source_reference.relative_path == "runbook.md"

    vector_repo.close()


@pytest.mark.parametrize(
    ("query", "max_results"),
    [("", 10), ("   ", 10), ("valid", 0), ("valid", MAX_RESULTS + 1)],
)
def test_vector_retriever_rejects_invalid_query_and_bounds(query: str, max_results: int) -> None:
    repo = SqliteRepository(":memory:", vector_index_enabled=True)
    provider = FakeEmbeddingProvider(dimension=4)
    retriever = VectorRetriever(provider, repo)

    with pytest.raises(InvalidQueryError):
        retriever.search(query, max_results=max_results)

    repo.close()
