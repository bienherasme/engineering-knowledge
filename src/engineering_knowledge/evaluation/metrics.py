"""Pure retrieval-quality metrics.

Neither function knows about SQLite, embeddings, or KnowledgeService: both
operate on plain ordered chunk_id sequences, so they can be tested with
hand-computable fixtures and reused by anything that produces a ranked
chunk_id list, not just this evaluation harness.
"""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Fraction of relevant chunk_ids present in the top k retrieved chunk_ids."""
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")
    relevant_set = set(relevant_ids)
    found = sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in relevant_set)
    return found / len(relevant_set)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """1 / rank of the first relevant chunk_id, or 0.0 if none appears."""
    relevant_set = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0
