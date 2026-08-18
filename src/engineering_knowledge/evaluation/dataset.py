"""Golden retrieval dataset: typed models, resolved through domain identity helpers.

Golden relevance judgments are stored as logical references (source_id,
relative_path, section_path, section_occurrence, ordinal_in_section), never
as raw chunk_id digests. The digest is a derived implementation detail of
the identity convention; hardcoding it in the dataset would make golden.json
both unreadable and silently stale the moment chunking behavior changes for
a reason nobody editing the JSON would notice. Resolution happens once, at
evaluation time, through the same derive_document_id/derive_chunk_id
helpers ingestion itself uses, so a golden reference that no longer points
at a real persisted chunk fails loudly instead of quietly evaluating
against the wrong, or no, content.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engineering_knowledge.domain import SectionPath
from engineering_knowledge.domain.identity import derive_chunk_id, derive_document_id
from engineering_knowledge.retrieval import KnowledgeRepository


class EvaluationDatasetError(ValueError):
    """The golden dataset is malformed, or a relevant reference does not resolve to a real chunk."""


class QueryCategory(StrEnum):
    """Diagnostic annotation of query character. Never affects ranking or metric weighting.

    lexical: expected to benefit from exact identifiers/terms.
    semantic: relevant content may use substantially different wording from the query.
    mixed: contains both exact engineering anchors and semantic intent.

    A lexical-category query is not guaranteed, or expected, to be won by
    lexical retrieval: the label describes the query's character, not a
    desired winner.
    """

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    MIXED = "mixed"


class RelevantChunkReference(BaseModel):
    """A logical, human-readable pointer to one relevant chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    relative_path: str
    section_path: tuple[str, ...]
    section_occurrence: int = Field(ge=0)
    ordinal_in_section: int = Field(ge=0)

    def resolve_chunk_id(self) -> str:
        document_id = derive_document_id(self.source_id, self.relative_path)
        section_identity = SectionPath(headings=self.section_path).as_identity_string()
        return derive_chunk_id(
            document_id, section_identity, self.section_occurrence, self.ordinal_in_section
        )


class GoldenQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    query: str
    category: QueryCategory
    relevant: tuple[RelevantChunkReference, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> GoldenQuery:
        if not self.query_id.strip():
            raise ValueError("query_id must not be blank")
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if not self.relevant:
            raise ValueError(f"{self.query_id}: must have at least one relevant reference")

        seen: set[tuple[str, str, tuple[str, ...], int, int]] = set()
        for ref in self.relevant:
            key = (
                ref.source_id,
                ref.relative_path,
                ref.section_path,
                ref.section_occurrence,
                ref.ordinal_in_section,
            )
            if key in seen:
                raise ValueError(f"{self.query_id}: duplicate relevant reference {key}")
            seen.add(key)
        return self


class GoldenDataset(BaseModel):
    """The frozen evaluation processing configuration travels with the dataset.

    ``max_chunk_chars`` here is not a suggestion: golden references were
    resolved against chunks produced under this exact bound, and evaluating
    against a corpus ingested with a different value would silently compare
    the dataset to chunk boundaries it was never actually judged against.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    max_chunk_chars: int = Field(gt=0)
    queries: tuple[GoldenQuery, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> GoldenDataset:
        if not self.queries:
            raise ValueError("dataset must contain queries")
        seen_ids: set[str] = set()
        for query in self.queries:
            if query.query_id in seen_ids:
                raise ValueError(f"duplicate query_id: {query.query_id}")
            seen_ids.add(query.query_id)
        return self

    @classmethod
    def load(cls, path: Path) -> GoldenDataset:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(payload)


def resolve_relevant_chunk_ids(
    dataset: GoldenDataset, repository: KnowledgeRepository
) -> dict[str, tuple[str, ...]]:
    """Resolve every query's logical references to chunk_ids, failing loudly on drift.

    Called once, before any strategy runs: a stale or missing golden
    reference is a dataset/corpus problem, not a "0 recall" result for that
    query, and silently dropping it would understate what the dataset
    actually claims to measure.
    """
    resolved: dict[str, tuple[str, ...]] = {}
    for query in dataset.queries:
        chunk_ids: list[str] = []
        for ref in query.relevant:
            chunk_id = ref.resolve_chunk_id()
            if repository.get_chunk(chunk_id) is None:
                raise EvaluationDatasetError(
                    f"{query.query_id}: relevant chunk_id {chunk_id!r} does not exist "
                    f"({ref.relative_path} {list(ref.section_path)}, "
                    f"occurrence={ref.section_occurrence}, ordinal={ref.ordinal_in_section})"
                )
            chunk_ids.append(chunk_id)
        resolved[query.query_id] = tuple(chunk_ids)
    return resolved
