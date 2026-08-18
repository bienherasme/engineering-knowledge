"""Hybrid retrieval: lexical and vector candidates fused by rank, not by score.

Reciprocal Rank Fusion combines a chunk's position in each retriever's
ranking, never BM25 and vector distance directly: those two scores live on
different, strategy-specific scales (lower-is-better FTS5 cost versus
lower-is-better nearest-neighbor distance, neither bounded the same way),
and adding or comparing them directly would be arithmetic over numbers that
do not mean the same thing. Rank is the only currency both retrievers
share honestly.

Hybrid retrieval requires both lexical and vector capability to actually
run. If the vector side is unavailable or incompatible, that failure
propagates rather than silently falling back to lexical-only: a caller who
asked for hybrid and got lexical-only without being told would draw
conclusions the system never actually computed. A caller who wants
lexical-only can already get it directly from ``LexicalIndex``.

Fusion itself (``fuse_rankings``) is a pure function over two match tuples:
no database, no provider, no query text. ``HybridRetriever`` is the only
piece that knows how to go from ordinary query text to those two tuples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from engineering_knowledge.domain import Chunk, SourceReference
from engineering_knowledge.retrieval.errors import HybridFusionError, InvalidQueryError
from engineering_knowledge.retrieval.lexical import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS,
    LexicalIndex,
    LexicalMatch,
)
from engineering_knowledge.retrieval.vector import VectorMatch, VectorRetriever

# A fixed initial fusion constant, not an evaluation-tuned value. 60 is the
# constant from the original Reciprocal Rank Fusion paper and is a common
# default; nothing here claims it is optimal for this corpus. The future
# evaluation harness is what would justify changing it.
RRF_K = 60

# How many candidates to request from each component relative to the final
# max_results, so RRF has more than the final page to find overlap in
# before truncating. Fixed and small on purpose: not a tunable knob yet.
_CANDIDATE_MULTIPLIER = 3


@dataclass(frozen=True, slots=True)
class HybridMatch:
    """One ranked hybrid result, fused from lexical and/or vector candidates.

    ``rank`` is this match's 1-based position in the final hybrid output.
    ``rrf_score`` is the fused score: higher is better, unlike the two
    native component scores below. ``lexical_rank``/``vector_rank`` are
    ``None`` when the chunk did not appear in that retriever's candidate
    list; same for ``bm25_score``/``vector_distance``, which are preserved
    exactly as their own retrievers report them (bm25_score lower-is-
    better, vector_distance lower-is-better) and are never combined with
    each other or with rrf_score.
    """

    chunk: Chunk
    source_reference: SourceReference
    rank: int
    rrf_score: float
    lexical_rank: int | None
    vector_rank: int | None
    bm25_score: float | None
    vector_distance: float | None


@dataclass
class _Candidate:
    """Mutable fusion accumulator; never exposed outside this module."""

    chunk: Chunk
    source_reference: SourceReference
    rrf_score: float = 0.0
    lexical_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    vector_distance: float | None = None


def fuse_rankings(
    lexical_matches: Sequence[LexicalMatch],
    vector_matches: Sequence[VectorMatch],
    *,
    max_results: int,
) -> tuple[HybridMatch, ...]:
    """Combine two rank-ordered candidate lists into one hybrid ranking.

    Candidates are identified strictly by chunk_id. A chunk present in both
    lists contributes two RRF terms and carries both components' rank and
    native score; a chunk present in only one contributes one term and
    leaves the other component's fields ``None``. Neither list needs to be
    sorted by this function; each match's own ``rank`` is trusted as-is.
    """
    candidates: dict[str, _Candidate] = {}

    seen_lexical: set[str] = set()
    for lexical_match in lexical_matches:
        chunk_id = lexical_match.chunk.chunk_id
        if chunk_id in seen_lexical:
            raise HybridFusionError(
                f"lexical component returned chunk_id {chunk_id!r} more than once"
            )
        seen_lexical.add(chunk_id)

        candidate = _get_or_create_candidate(
            candidates, chunk_id, lexical_match.chunk, lexical_match.source_reference
        )
        candidate.rrf_score += 1.0 / (RRF_K + lexical_match.rank)
        candidate.lexical_rank = lexical_match.rank
        candidate.bm25_score = lexical_match.bm25_score

    seen_vector: set[str] = set()
    for vector_match in vector_matches:
        chunk_id = vector_match.chunk.chunk_id
        if chunk_id in seen_vector:
            raise HybridFusionError(
                f"vector component returned chunk_id {chunk_id!r} more than once"
            )
        seen_vector.add(chunk_id)

        candidate = _get_or_create_candidate(
            candidates, chunk_id, vector_match.chunk, vector_match.source_reference
        )
        candidate.rrf_score += 1.0 / (RRF_K + vector_match.rank)
        candidate.vector_rank = vector_match.rank
        candidate.vector_distance = vector_match.distance

    ordered = sorted(candidates.values(), key=_tie_break_key)
    return tuple(
        HybridMatch(
            chunk=candidate.chunk,
            source_reference=candidate.source_reference,
            rank=rank,
            rrf_score=candidate.rrf_score,
            lexical_rank=candidate.lexical_rank,
            vector_rank=candidate.vector_rank,
            bm25_score=candidate.bm25_score,
            vector_distance=candidate.vector_distance,
        )
        for rank, candidate in enumerate(ordered[:max_results], start=1)
    )


def _get_or_create_candidate(
    candidates: dict[str, _Candidate],
    chunk_id: str,
    chunk: Chunk,
    source_reference: SourceReference,
) -> _Candidate:
    existing = candidates.get(chunk_id)
    if existing is None:
        candidate = _Candidate(chunk=chunk, source_reference=source_reference)
        candidates[chunk_id] = candidate
        return candidate

    # Both retrievers resolve from the same normalized tables, so this
    # should always hold; checking content_hash rather than trusting it
    # catches a genuinely inconsistent read instead of silently mixing two
    # different versions of the same chunk_id into one result.
    if existing.chunk.content_hash != chunk.content_hash:
        raise HybridFusionError(
            f"lexical and vector components disagree on content_hash for "
            f"chunk_id {chunk_id!r}"
        )
    return existing


_MISSING_RANK = float("inf")


def _tie_break_key(candidate: _Candidate) -> tuple[float, float, float, float, str, int, str]:
    ranks = [r for r in (candidate.lexical_rank, candidate.vector_rank) if r is not None]
    best_rank = float(min(ranks)) if ranks else _MISSING_RANK
    lexical_rank = (
        float(candidate.lexical_rank) if candidate.lexical_rank is not None else _MISSING_RANK
    )
    vector_rank = (
        float(candidate.vector_rank) if candidate.vector_rank is not None else _MISSING_RANK
    )
    return (
        -candidate.rrf_score,
        best_rank,
        lexical_rank,
        vector_rank,
        candidate.source_reference.relative_path,
        candidate.chunk.ordinal,
        candidate.chunk.chunk_id,
    )


class HybridRetriever:
    """Composes LexicalIndex and VectorRetriever, fusing both by rank.

    Depends only on the existing ports: it never touches SQL, FTS syntax,
    sqlite-vec, or an EmbeddingProvider directly. The same ordinary query
    text is passed unchanged to both components; each applies its own
    query transformation internally, and this class does not tokenize,
    lowercase, or rewrite anything.
    """

    def __init__(self, lexical_index: LexicalIndex, vector_retriever: VectorRetriever) -> None:
        self._lexical_index = lexical_index
        self._vector_retriever = vector_retriever

    def search(
        self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS
    ) -> tuple[HybridMatch, ...]:
        stripped = query.strip()
        if not stripped:
            raise InvalidQueryError("query must not be blank")
        if max_results <= 0:
            raise InvalidQueryError("max_results must be positive")
        if max_results > MAX_RESULTS:
            raise InvalidQueryError(f"max_results must not exceed {MAX_RESULTS}")

        candidate_limit = min(max_results * _CANDIDATE_MULTIPLIER, MAX_RESULTS)

        lexical_matches = self._lexical_index.search(stripped, max_results=candidate_limit)
        # No try/except here: an unavailable or incompatible vector index
        # must fail this call outright, not degrade hybrid into lexical
        # search without saying so.
        vector_matches = self._vector_retriever.search(stripped, max_results=candidate_limit)

        return fuse_rankings(lexical_matches, vector_matches, max_results=max_results)
