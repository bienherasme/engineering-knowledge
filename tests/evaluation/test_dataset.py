from pathlib import Path

import pytest

from engineering_knowledge.domain import Chunk, Document, SectionPath
from engineering_knowledge.domain.identity import derive_chunk_id, derive_document_id
from engineering_knowledge.evaluation.dataset import (
    EvaluationDatasetError,
    GoldenDataset,
    GoldenQuery,
    QueryCategory,
    RelevantChunkReference,
    resolve_relevant_chunk_ids,
)
from engineering_knowledge.ingestion import IngestionService
from engineering_knowledge.persistence import SqliteRepository
from engineering_knowledge.sources import LocalFilesystemSourceAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "corpus"
GOLDEN_DATASET_PATH = REPO_ROOT / "eval" / "golden_dataset.json"


class _FakeRepository:
    """A minimal KnowledgeRepository double that knows about no chunks."""

    def get_document(self, document_id: str) -> Document | None:
        raise AssertionError("get_document must not be called by resolve_relevant_chunk_ids")

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return None


def test_relevant_chunk_reference_resolves_via_domain_identity_helpers() -> None:
    reference = RelevantChunkReference(
        source_id="docs",
        relative_path="runbook.md",
        section_path=("Payments", "Rollback"),
        section_occurrence=0,
        ordinal_in_section=1,
    )

    expected_document_id = derive_document_id("docs", "runbook.md")
    expected_chunk_id = derive_chunk_id(
        expected_document_id,
        SectionPath(headings=("Payments", "Rollback")).as_identity_string(),
        section_occurrence=0,
        ordinal_in_section=1,
    )

    assert reference.resolve_chunk_id() == expected_chunk_id


def test_resolve_relevant_chunk_ids_raises_on_stale_reference() -> None:
    dataset = GoldenDataset(
        source_id="docs",
        max_chunk_chars=1000,
        queries=(
            GoldenQuery(
                query_id="q1",
                query="anything",
                category=QueryCategory.LEXICAL,
                relevant=(
                    RelevantChunkReference(
                        source_id="docs",
                        relative_path="missing.md",
                        section_path=("Nowhere",),
                        section_occurrence=0,
                        ordinal_in_section=0,
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(EvaluationDatasetError):
        resolve_relevant_chunk_ids(dataset, _FakeRepository())


def test_golden_dataset_resolves_against_real_corpus(tmp_path: Path) -> None:
    dataset = GoldenDataset.load(GOLDEN_DATASET_PATH)

    repo = SqliteRepository(str(tmp_path / "eval.db"))
    service = IngestionService(repo, max_chunk_chars=dataset.max_chunk_chars)
    service.ingest_source(
        LocalFilesystemSourceAdapter(source_id=dataset.source_id, root=CORPUS_DIR)
    )

    resolved = resolve_relevant_chunk_ids(dataset, repo)

    assert len(resolved) == len(dataset.queries)
    assert all(chunk_ids for chunk_ids in resolved.values())

    repo.close()
