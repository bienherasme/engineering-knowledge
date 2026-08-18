from engineering_knowledge.domain import Chunk, Document, SectionPath, SourceReference
from engineering_knowledge.domain.identity import derive_document_id
from engineering_knowledge.evaluation.dataset import (
    GoldenDataset,
    GoldenQuery,
    QueryCategory,
    RelevantChunkReference,
)
from engineering_knowledge.evaluation.runner import run_evaluation
from engineering_knowledge.retrieval import (
    RetrievalHit,
    RetrievalResult,
    RetrievalStatus,
    RetrievalStrategy,
)


def _make_chunk(document_id: str, text: str) -> Chunk:
    return Chunk.create(
        document_id=document_id,
        section_path=SectionPath(),
        section_occurrence=0,
        ordinal=0,
        ordinal_in_section=0,
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


def _make_hit(chunk: Chunk, relative_path: str, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk=chunk,
        source_reference=_make_reference(chunk, relative_path),
        rank=rank,
        lexical_rank=rank,
        bm25_score=-float(rank),
    )


class _FakeKnowledgeService:
    """A tiny KnowledgeService-shaped double: fixed per-query search results."""

    def __init__(
        self, chunks_by_id: dict[str, Chunk], results_by_query: dict[str, RetrievalResult]
    ) -> None:
        self._chunks_by_id = chunks_by_id
        self._results_by_query = results_by_query

    def get_document(self, document_id: str) -> Document | None:
        raise AssertionError("get_document must not be called by run_evaluation")

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks_by_id.get(chunk_id)

    def search(
        self,
        query: str,
        *,
        strategy: RetrievalStrategy | None = None,
        max_results: int = 5,
    ) -> RetrievalResult:
        return self._results_by_query[query]


def _build_dataset_and_service() -> tuple[GoldenDataset, _FakeKnowledgeService]:
    # document_ids are derived exactly as RelevantChunkReference.resolve_chunk_id()
    # derives them, so these fixtures naturally share chunk_id with the
    # dataset's logical references below without any post-hoc patching.
    hit_document_id = derive_document_id("docs", "a.md")
    missed_document_id = derive_document_id("docs", "c.md")

    hit_chunk = _make_chunk(hit_document_id, "the relevant content")
    distractor_chunk = _make_chunk("doc_distractor", "unrelated content")
    missed_chunk = _make_chunk(missed_document_id, "never retrieved content")

    dataset = GoldenDataset(
        source_id="docs",
        max_chunk_chars=1000,
        queries=(
            GoldenQuery(
                query_id="q1-first-rank-hit",
                query="find it now",
                category=QueryCategory.LEXICAL,
                relevant=(
                    RelevantChunkReference(
                        source_id="docs",
                        relative_path="a.md",
                        section_path=hit_chunk.section_path.headings,
                        section_occurrence=0,
                        ordinal_in_section=0,
                    ),
                ),
            ),
            GoldenQuery(
                query_id="q2-second-rank-hit",
                query="find it eventually",
                category=QueryCategory.SEMANTIC,
                relevant=(
                    RelevantChunkReference(
                        source_id="docs",
                        relative_path="a.md",
                        section_path=hit_chunk.section_path.headings,
                        section_occurrence=0,
                        ordinal_in_section=0,
                    ),
                ),
            ),
            GoldenQuery(
                query_id="q3-never-found",
                query="never find it",
                category=QueryCategory.MIXED,
                relevant=(
                    RelevantChunkReference(
                        source_id="docs",
                        relative_path="c.md",
                        section_path=missed_chunk.section_path.headings,
                        section_occurrence=0,
                        ordinal_in_section=0,
                    ),
                ),
            ),
        ),
    )

    assert hit_chunk.chunk_id == dataset.queries[0].relevant[0].resolve_chunk_id()
    assert missed_chunk.chunk_id == dataset.queries[2].relevant[0].resolve_chunk_id()

    results_by_query = {
        "find it now": RetrievalResult(
            strategy=RetrievalStrategy.LEXICAL,
            status=RetrievalStatus.SUCCESS,
            results=(_make_hit(hit_chunk, "a.md", 1),),
            requested_max_results=5,
            truncated=False,
        ),
        "find it eventually": RetrievalResult(
            strategy=RetrievalStrategy.LEXICAL,
            status=RetrievalStatus.SUCCESS,
            results=(
                _make_hit(distractor_chunk, "b.md", 1),
                _make_hit(hit_chunk, "a.md", 2),
            ),
            requested_max_results=5,
            truncated=False,
        ),
        "never find it": RetrievalResult(
            strategy=RetrievalStrategy.LEXICAL,
            status=RetrievalStatus.SUCCESS,
            results=(_make_hit(distractor_chunk, "b.md", 1),),
            requested_max_results=5,
            truncated=False,
        ),
    }

    chunks_by_id = {
        hit_chunk.chunk_id: hit_chunk,
        distractor_chunk.chunk_id: distractor_chunk,
        missed_chunk.chunk_id: missed_chunk,
    }
    service = _FakeKnowledgeService(chunks_by_id, results_by_query)
    return dataset, service


def test_run_evaluation_aggregates_recall_and_mrr_across_queries() -> None:
    dataset, service = _build_dataset_and_service()

    report = run_evaluation(service, dataset, strategies=(RetrievalStrategy.LEXICAL,), k=5)

    assert report.k == 5
    assert report.total_queries == 3
    assert len(report.strategies) == 1

    strategy_eval = report.strategies[0]
    assert strategy_eval.strategy == RetrievalStrategy.LEXICAL
    assert strategy_eval.query_count == 3
    assert strategy_eval.mean_recall_at_k == (1.0 + 1.0 + 0.0) / 3
    assert strategy_eval.mrr == (1.0 + 0.5 + 0.0) / 3

    by_query_id = {qe.query_id: qe for qe in strategy_eval.query_results}
    assert by_query_id["q1-first-rank-hit"].recall_at_k == 1.0
    assert by_query_id["q1-first-rank-hit"].reciprocal_rank == 1.0
    assert by_query_id["q2-second-rank-hit"].reciprocal_rank == 0.5
    assert by_query_id["q3-never-found"].recall_at_k == 0.0
    assert by_query_id["q3-never-found"].reciprocal_rank == 0.0


def test_run_evaluation_category_breakdown_isolates_each_category() -> None:
    dataset, service = _build_dataset_and_service()

    report = run_evaluation(service, dataset, strategies=(RetrievalStrategy.LEXICAL,), k=5)
    breakdown_by_category = {cb.category: cb for cb in report.strategies[0].category_breakdown}

    assert breakdown_by_category[QueryCategory.LEXICAL].query_count == 1
    assert breakdown_by_category[QueryCategory.LEXICAL].mean_recall_at_k == 1.0
    assert breakdown_by_category[QueryCategory.LEXICAL].mrr == 1.0

    assert breakdown_by_category[QueryCategory.SEMANTIC].query_count == 1
    assert breakdown_by_category[QueryCategory.SEMANTIC].mrr == 0.5

    assert breakdown_by_category[QueryCategory.MIXED].query_count == 1
    assert breakdown_by_category[QueryCategory.MIXED].mean_recall_at_k == 0.0
    assert breakdown_by_category[QueryCategory.MIXED].mrr == 0.0
