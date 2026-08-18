from datetime import UTC, datetime
from pathlib import Path

import pytest

from engineering_knowledge.domain import Chunk, SectionPath, SourceReference
from engineering_knowledge.embeddings import FakeEmbeddingProvider
from engineering_knowledge.ingestion import IngestionService
from engineering_knowledge.persistence import SqliteRepository
from engineering_knowledge.retrieval import (
    InvalidQueryError,
    KnowledgeService,
    LexicalMatch,
    RetrievalStatus,
    RetrievalStrategy,
    VectorIndexUnavailableError,
    VectorMatch,
    VectorRetriever,
)
from engineering_knowledge.sources import LocalFilesystemSourceAdapter


def _make_chunk(document_id: str, ordinal: int, text: str) -> Chunk:
    return Chunk.create(
        document_id=document_id,
        section_path=SectionPath(),
        section_occurrence=0,
        ordinal=ordinal,
        ordinal_in_section=ordinal,
        text=text,
    )


def _make_reference(chunk: Chunk, relative_path: str) -> SourceReference:
    return SourceReference(
        source_id="docs",
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        relative_path=relative_path,
        section_path=chunk.section_path,
        section_occurrence=chunk.section_occurrence,
        content_hash=chunk.content_hash,
    )


class _FixedLexicalIndex:
    """Returns a fixed, already-ranked match list, truncated like a real index would."""

    def __init__(self, matches: tuple[LexicalMatch, ...]) -> None:
        self._matches = matches

    def search(self, query: str, *, max_results: int = 10) -> tuple[LexicalMatch, ...]:
        return self._matches[:max_results]


class _UnreachableLexicalIndex:
    def search(self, query: str, *, max_results: int = 10) -> tuple[LexicalMatch, ...]:
        raise AssertionError("lexical search must not run for invalid input")


class _UnreachableVectorRetriever:
    def search(self, query: str, *, max_results: int = 10) -> tuple[VectorMatch, ...]:
        raise AssertionError("vector search/embedding call must not run for invalid input")


def test_search_lexical_success_and_empty(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "runbook.md").write_text("# Payments\n\nRollback steps for Aegis.\n")

    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000, clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=root))

    knowledge = KnowledgeService(repo, repo)

    success = knowledge.search("rollback")
    assert success.status == RetrievalStatus.SUCCESS
    assert success.strategy == RetrievalStrategy.LEXICAL
    assert len(success.results) == 1
    hit = success.results[0]
    assert hit.rank == 1
    assert hit.lexical_rank == 1
    assert hit.bm25_score is not None
    assert hit.vector_rank is None
    assert hit.rrf_score is None
    assert hit.source_reference.chunk_id == hit.chunk.chunk_id
    assert success.truncated is False
    assert success.truncation_reason is None

    empty = knowledge.search("nonexistent_term_zzz")
    assert empty.status == RetrievalStatus.EMPTY
    assert empty.results == ()
    assert empty.truncated is False
    assert empty.truncation_reason is None

    repo.close()


def test_search_truncation_probe_marks_partial_and_exposes_only_requested_count() -> None:
    chunk1 = _make_chunk("doc_1", 0, "content one")
    chunk2 = _make_chunk("doc_2", 0, "content two")
    chunk3 = _make_chunk("doc_3", 0, "content three")
    matches = (
        LexicalMatch(
            chunk=chunk1,
            source_reference=_make_reference(chunk1, "one.md"),
            rank=1,
            bm25_score=-3.0,
        ),
        LexicalMatch(
            chunk=chunk2,
            source_reference=_make_reference(chunk2, "two.md"),
            rank=2,
            bm25_score=-2.0,
        ),
        LexicalMatch(
            chunk=chunk3,
            source_reference=_make_reference(chunk3, "three.md"),
            rank=3,
            bm25_score=-1.0,
        ),
    )
    knowledge = KnowledgeService(SqliteRepository(":memory:"), _FixedLexicalIndex(matches))

    result = knowledge.search("anything", max_results=2)

    assert result.status == RetrievalStatus.PARTIAL
    assert len(result.results) == 2
    assert [hit.rank for hit in result.results] == [1, 2]
    assert result.truncated is True
    assert result.truncation_reason == "max_results"


def test_search_strategy_dispatch_shapes_component_fields(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "runbook.md").write_text(
        "# Payments\n\nRollback steps for Aegis payments deployment.\n"
    )

    provider = FakeEmbeddingProvider(dimension=4)
    repo = SqliteRepository(":memory:", vector_index_enabled=True)
    service = IngestionService(
        repo, max_chunk_chars=1000, embedding_provider=provider,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=root))
    repo.rebuild_vector_index(provider)

    knowledge = KnowledgeService(repo, repo, VectorRetriever(provider, repo))

    for strategy in (RetrievalStrategy.LEXICAL, RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID):
        result = knowledge.search("payments deployment", strategy=strategy, max_results=5)
        assert result.strategy == strategy
        hit = result.results[0]

        if strategy is RetrievalStrategy.LEXICAL:
            assert hit.lexical_rank is not None and hit.bm25_score is not None
            assert hit.vector_rank is None and hit.vector_distance is None and hit.rrf_score is None
        elif strategy is RetrievalStrategy.VECTOR:
            assert hit.vector_rank is not None and hit.vector_distance is not None
            assert hit.lexical_rank is None and hit.bm25_score is None and hit.rrf_score is None
        else:
            assert hit.rrf_score is not None
            assert hit.lexical_rank is not None and hit.bm25_score is not None
            assert hit.vector_rank is not None and hit.vector_distance is not None

    repo.close()


@pytest.mark.parametrize("strategy", [RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID])
def test_search_vector_and_hybrid_without_configured_retriever_fail_explicitly(
    tmp_path: Path, strategy: RetrievalStrategy
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "runbook.md").write_text("# Payments\n\nRollback steps.\n")

    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000, clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=root))

    knowledge = KnowledgeService(repo, repo)  # no vector_retriever configured

    with pytest.raises(VectorIndexUnavailableError):
        knowledge.search("rollback", strategy=strategy)

    # no silent fallback: lexical still answers the same query directly
    lexical_result = knowledge.search("rollback", strategy=RetrievalStrategy.LEXICAL)
    assert lexical_result.status == RetrievalStatus.SUCCESS

    repo.close()


@pytest.mark.parametrize(("query", "max_results"), [("", 10), ("valid query", 0)])
def test_search_rejects_invalid_input_before_any_retrieval_work(
    query: str, max_results: int
) -> None:
    knowledge = KnowledgeService(
        SqliteRepository(":memory:"), _UnreachableLexicalIndex(), _UnreachableVectorRetriever()  # type: ignore[arg-type]
    )

    with pytest.raises(InvalidQueryError):
        knowledge.search(query, strategy=RetrievalStrategy.HYBRID, max_results=max_results)


def test_get_document_and_get_chunk_delegate_to_repository(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "runbook.md").write_text("# Payments\n\nRollback steps.\n")

    repo = SqliteRepository(":memory:")
    service = IngestionService(
        repo, max_chunk_chars=1000, clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=root))

    knowledge = KnowledgeService(repo, repo)

    document = repo.list_documents_for_source("docs")[0]
    chunk = repo.get_chunks(document.document_id)[0]

    assert knowledge.get_document(document.document_id) == document
    assert knowledge.get_chunk(chunk.chunk_id) == chunk
    assert knowledge.get_document("doc_nonexistent") is None
    assert knowledge.get_chunk("chunk_nonexistent") is None

    repo.close()
