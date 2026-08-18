from datetime import UTC, datetime
from pathlib import Path

import pytest

from engineering_knowledge.domain import Chunk, SectionPath, SourceReference

# vector_index_enabled=True loads the optional sqlite-vec extension; skip
# this whole module cleanly in a base-only environment rather than failing.
pytest.importorskip("sqlite_vec")

from engineering_knowledge.embeddings import FakeEmbeddingProvider
from engineering_knowledge.ingestion import IngestionService
from engineering_knowledge.persistence import SqliteRepository
from engineering_knowledge.retrieval import (
    MAX_RESULTS,
    RRF_K,
    HybridFusionError,
    HybridRetriever,
    InvalidQueryError,
    LexicalMatch,
    VectorMatch,
    VectorRetriever,
    fuse_rankings,
)
from engineering_knowledge.retrieval.errors import VectorIndexUnavailableError
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


def test_fuse_rankings_overlap_candidate_outranks_single_component_candidates() -> None:
    chunk_a = _make_chunk("doc_a", 0, "content A")
    chunk_b = _make_chunk("doc_b", 0, "content B")
    chunk_c = _make_chunk("doc_c", 0, "content C")
    ref_a, ref_b, ref_c = (
        _make_reference(chunk_a, "a.md"),
        _make_reference(chunk_b, "b.md"),
        _make_reference(chunk_c, "c.md"),
    )

    lexical = (
        LexicalMatch(chunk=chunk_a, source_reference=ref_a, rank=1, bm25_score=-2.0),
        LexicalMatch(chunk=chunk_b, source_reference=ref_b, rank=2, bm25_score=-1.0),
    )
    vector = (
        VectorMatch(chunk=chunk_b, source_reference=ref_b, rank=1, distance=0.1),
        VectorMatch(chunk=chunk_c, source_reference=ref_c, rank=2, distance=0.5),
    )

    results = fuse_rankings(lexical, vector, max_results=10)

    assert [m.chunk.chunk_id for m in results] == [
        chunk_b.chunk_id,
        chunk_a.chunk_id,
        chunk_c.chunk_id,
    ]
    assert results[0].rrf_score == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert results[0].lexical_rank == 2
    assert results[0].vector_rank == 1
    assert results[0].bm25_score == -1.0
    assert results[0].vector_distance == 0.1

    # single-component candidates keep the other component's fields empty,
    # never a sentinel like 0.0
    assert results[1].vector_rank is None
    assert results[1].vector_distance is None
    assert results[2].lexical_rank is None
    assert results[2].bm25_score is None


def test_fuse_rankings_deterministic_tie_break() -> None:
    chunk_lexical_only = _make_chunk("doc_a", 0, "content A")
    chunk_vector_only = _make_chunk("doc_b", 0, "content B")
    # deliberately give the vector-only candidate the alphabetically first
    # path, so a naive relative_path-first assumption would get this wrong:
    # lexical_rank presence is checked before relative_path in the
    # documented tie-break order.
    ref_lexical_only = _make_reference(chunk_lexical_only, "z_lexical.md")
    ref_vector_only = _make_reference(chunk_vector_only, "a_vector.md")

    lexical = (
        LexicalMatch(
            chunk=chunk_lexical_only, source_reference=ref_lexical_only, rank=1, bm25_score=-1.0
        ),
    )
    vector = (
        VectorMatch(
            chunk=chunk_vector_only, source_reference=ref_vector_only, rank=1, distance=0.2
        ),
    )

    first = fuse_rankings(lexical, vector, max_results=10)
    second = fuse_rankings(lexical, vector, max_results=10)

    assert first == second
    assert first[0].rrf_score == pytest.approx(first[1].rrf_score)
    assert first[0].chunk.chunk_id == chunk_lexical_only.chunk_id
    assert first[1].chunk.chunk_id == chunk_vector_only.chunk_id


@pytest.mark.parametrize("duplicate_side", ["lexical", "vector"])
def test_fuse_rankings_rejects_duplicate_chunk_id_from_one_component(duplicate_side: str) -> None:
    chunk = _make_chunk("doc_a", 0, "content A")
    reference = _make_reference(chunk, "a.md")

    lexical: tuple[LexicalMatch, ...] = ()
    vector: tuple[VectorMatch, ...] = ()
    if duplicate_side == "lexical":
        lexical = (
            LexicalMatch(chunk=chunk, source_reference=reference, rank=1, bm25_score=-1.0),
            LexicalMatch(chunk=chunk, source_reference=reference, rank=2, bm25_score=-0.5),
        )
    else:
        vector = (
            VectorMatch(chunk=chunk, source_reference=reference, rank=1, distance=0.1),
            VectorMatch(chunk=chunk, source_reference=reference, rank=2, distance=0.3),
        )

    with pytest.raises(HybridFusionError):
        fuse_rankings(lexical, vector, max_results=10)


class _UnreachableLexicalIndex:
    def search(self, query: str, *, max_results: int = 10) -> tuple[LexicalMatch, ...]:
        raise AssertionError("lexical search must not run for invalid input")


class _UnreachableVectorRetriever:
    def search(self, query: str, *, max_results: int = 10) -> tuple[VectorMatch, ...]:
        raise AssertionError("vector search must not run for invalid input")


@pytest.mark.parametrize(
    ("query", "max_results"),
    [("", 10), ("   ", 10), ("valid", 0), ("valid", MAX_RESULTS + 1)],
)
def test_hybrid_retriever_rejects_invalid_input_before_any_retrieval_work(
    query: str, max_results: int
) -> None:
    hybrid = HybridRetriever(_UnreachableLexicalIndex(), _UnreachableVectorRetriever())  # type: ignore[arg-type]

    with pytest.raises(InvalidQueryError):
        hybrid.search(query, max_results=max_results)


def test_hybrid_retriever_integration_with_real_components(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "runbook.md").write_text(
        "# Payments\n\nRollback steps for Aegis payments deployment.\n"
    )
    (root / "config.txt").write_text("MAX_RETRY_COUNT=5\npayments-service timeout settings.\n")

    provider = FakeEmbeddingProvider(dimension=4)
    repo = SqliteRepository(":memory:", vector_index_enabled=True)
    service = IngestionService(
        repo, max_chunk_chars=1000, embedding_provider=provider,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=root))
    repo.rebuild_vector_index(provider)

    vector_retriever = VectorRetriever(provider, repo)
    hybrid = HybridRetriever(repo, vector_retriever)

    results = hybrid.search("payments deployment", max_results=5)

    assert len(results) <= 5
    chunk_ids = [match.chunk.chunk_id for match in results]
    assert len(chunk_ids) == len(set(chunk_ids))
    for match in results:
        assert match.rank >= 1
        assert match.source_reference.chunk_id == match.chunk.chunk_id
        assert match.source_reference.content_hash == match.chunk.content_hash
        assert match.lexical_rank is not None or match.vector_rank is not None
        assert match.rrf_score > 0

    repo.close()


def test_hybrid_retriever_propagates_vector_unavailable(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "runbook.md").write_text("# Payments\n\nRollback steps.\n")

    repo = SqliteRepository(":memory:", vector_index_enabled=True)
    service = IngestionService(
        repo, max_chunk_chars=1000, clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    service.ingest_source(LocalFilesystemSourceAdapter(source_id="docs", root=root))
    # no rebuild_vector_index call: vector capability stays unbuilt

    assert len(repo.search("rollback")) == 1

    provider = FakeEmbeddingProvider(dimension=4)
    hybrid = HybridRetriever(repo, VectorRetriever(provider, repo))

    with pytest.raises(VectorIndexUnavailableError):
        hybrid.search("rollback")

    repo.close()
