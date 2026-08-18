"""Evaluation harness: runs the public KnowledgeService across strategies and aggregates metrics.

Evaluates KnowledgeService.search, not the internal retrievers directly:
the claim this harness produces evidence for is what a real consumer gets
through the stable public retrieval contract, strategy composition
included, not what an internal component could theoretically do in
isolation.

PARTIAL results are ordinary evidence, not a harness failure: metrics use
exactly result.results, the public top k. An infrastructure error (an
unavailable or incompatible vector index, a lexical index failure) is never
caught here and converted into a zero score; it propagates and fails the
evaluation run, since a zero score would misrepresent "the system could not
answer" as "the system answered and found nothing relevant".
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from engineering_knowledge.evaluation.dataset import (
    GoldenDataset,
    GoldenQuery,
    QueryCategory,
    resolve_relevant_chunk_ids,
)
from engineering_knowledge.evaluation.metrics import recall_at_k, reciprocal_rank
from engineering_knowledge.retrieval import KnowledgeService, RetrievalStrategy

DEFAULT_STRATEGIES: tuple[RetrievalStrategy, ...] = (
    RetrievalStrategy.LEXICAL,
    RetrievalStrategy.VECTOR,
    RetrievalStrategy.HYBRID,
)
DEFAULT_K = 5


class EvaluationRunnerError(ValueError):
    """The runner observed a retrieval-contract violation, such as duplicate retrieved chunk_ids."""


class QueryEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    category: QueryCategory
    strategy: RetrievalStrategy
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    recall_at_k: float
    reciprocal_rank: float


class CategoryBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: QueryCategory
    query_count: int = Field(ge=0)
    mean_recall_at_k: float
    mrr: float


class StrategyEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: RetrievalStrategy
    query_count: int = Field(ge=0)
    mean_recall_at_k: float
    mrr: float
    category_breakdown: tuple[CategoryBreakdown, ...]
    query_results: tuple[QueryEvaluation, ...]


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int = Field(gt=0)
    total_queries: int = Field(ge=0)
    strategies: tuple[StrategyEvaluation, ...]


def run_evaluation(
    knowledge_service: KnowledgeService,
    dataset: GoldenDataset,
    *,
    strategies: Sequence[RetrievalStrategy] = DEFAULT_STRATEGIES,
    k: int = DEFAULT_K,
) -> EvaluationReport:
    if k <= 0:
        raise ValueError("k must be positive")

    # KnowledgeService.get_document/get_chunk already structurally satisfy
    # KnowledgeRepository, so no separate repository handle is needed here.
    relevant_by_query = resolve_relevant_chunk_ids(dataset, knowledge_service)

    strategy_evaluations = [
        _evaluate_strategy(knowledge_service, dataset, relevant_by_query, strategy, k)
        for strategy in strategies
    ]

    return EvaluationReport(
        k=k, total_queries=len(dataset.queries), strategies=tuple(strategy_evaluations)
    )


def _evaluate_strategy(
    knowledge_service: KnowledgeService,
    dataset: GoldenDataset,
    relevant_by_query: dict[str, tuple[str, ...]],
    strategy: RetrievalStrategy,
    k: int,
) -> StrategyEvaluation:
    query_evaluations = [
        _evaluate_query(knowledge_service, query, relevant_by_query[query.query_id], strategy, k)
        for query in dataset.queries
    ]

    category_breakdown = tuple(
        _category_breakdown(category, query_evaluations) for category in QueryCategory
    )

    return StrategyEvaluation(
        strategy=strategy,
        query_count=len(query_evaluations),
        mean_recall_at_k=mean(qe.recall_at_k for qe in query_evaluations),
        mrr=mean(qe.reciprocal_rank for qe in query_evaluations),
        category_breakdown=category_breakdown,
        query_results=tuple(query_evaluations),
    )


def _evaluate_query(
    knowledge_service: KnowledgeService,
    query: GoldenQuery,
    relevant_ids: tuple[str, ...],
    strategy: RetrievalStrategy,
    k: int,
) -> QueryEvaluation:
    result = knowledge_service.search(query.query, strategy=strategy, max_results=k)
    retrieved_ids = tuple(hit.chunk.chunk_id for hit in result.results)

    if len(set(retrieved_ids)) != len(retrieved_ids):
        raise EvaluationRunnerError(
            f"{query.query_id}/{strategy}: retrieved duplicate chunk_ids: {retrieved_ids}"
        )

    return QueryEvaluation(
        query_id=query.query_id,
        category=query.category,
        strategy=strategy,
        relevant_chunk_ids=relevant_ids,
        retrieved_chunk_ids=retrieved_ids,
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
        reciprocal_rank=reciprocal_rank(retrieved_ids, relevant_ids),
    )


def _category_breakdown(
    category: QueryCategory, query_evaluations: list[QueryEvaluation]
) -> CategoryBreakdown:
    subset = [qe for qe in query_evaluations if qe.category is category]
    return CategoryBreakdown(
        category=category,
        query_count=len(subset),
        mean_recall_at_k=mean(qe.recall_at_k for qe in subset) if subset else 0.0,
        mrr=mean(qe.reciprocal_rank for qe in subset) if subset else 0.0,
    )
