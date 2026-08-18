from engineering_knowledge.evaluation.metrics import recall_at_k, reciprocal_rank


def test_recall_at_k_counts_relevant_ids_within_top_k() -> None:
    retrieved = ("chunk_a", "chunk_b", "chunk_c", "chunk_d")
    relevant = ("chunk_c", "chunk_z")

    assert recall_at_k(retrieved, relevant, k=2) == 0.0
    assert recall_at_k(retrieved, relevant, k=3) == 0.5
    assert recall_at_k(retrieved, relevant, k=4) == 0.5
    assert recall_at_k((), relevant, k=5) == 0.0


def test_reciprocal_rank_returns_inverse_of_first_relevant_rank() -> None:
    relevant = ("chunk_b", "chunk_c")

    assert reciprocal_rank(("chunk_a", "chunk_b", "chunk_c"), relevant) == 0.5
    assert reciprocal_rank(("chunk_b", "chunk_a"), relevant) == 1.0
    assert reciprocal_rank(("chunk_x", "chunk_y"), relevant) == 0.0
    assert reciprocal_rank((), relevant) == 0.0
