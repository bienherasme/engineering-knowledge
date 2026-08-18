"""Public retrieval surface: the one boundary a consumer should need.

``KnowledgeService`` hides LexicalIndex, VectorRetriever, HybridRetriever,
BM25, sqlite-vec, and RRF composition behind ``search(...) -> RetrievalResult``
plus two small identity reads. The internal strategy-specific ports and
match models are not replaced by this: they remain the right boundary for
evaluation and testing. This is the layer above them a normal caller uses.

``RetrievalResult.status`` and raised exceptions are two different axes on
purpose. SUCCESS/EMPTY/PARTIAL describe the shape of a *valid* retrieval
outcome; a blank query, an unavailable index, or a component contract
violation are never folded into a status value, they propagate as the
existing typed errors (``InvalidQueryError``, ``IndexUnavailableError``,
``VectorIndexUnavailableError``, ``VectorIndexIncompatibleError``,
``HybridFusionError``). Converting any of those into a result status would
erase the distinction between "here is a valid, possibly truncated answer"
and "the question or the system could not be answered at all".

PARTIAL means the public max_results bound is known to have truncated
further ranked results, nothing more. A vector or hybrid query over an
index with more than max_results chunks will routinely come back PARTIAL,
since nearest-neighbor search returns neighbors regardless of how weak the
match is; that is expected truncation bookkeeping, not degraded
infrastructure or low answer quality, and no similarity threshold is
introduced here to change that.

The service's own default strategy is lexical: base installs work without
the vector extras or an active vector index, and a default that silently
changed based on which optional packages happen to be installed would be
much harder to reason about than one that is always explicit.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engineering_knowledge.domain import Chunk, Document, SourceReference
from engineering_knowledge.retrieval.errors import InvalidQueryError, VectorIndexUnavailableError
from engineering_knowledge.retrieval.hybrid import HybridMatch, HybridRetriever
from engineering_knowledge.retrieval.lexical import LexicalIndex, LexicalMatch
from engineering_knowledge.retrieval.vector import VectorMatch, VectorRetriever

DEFAULT_PUBLIC_MAX_RESULTS = 10
# Deliberately below the internal retrievers' MAX_RESULTS (50): requesting
# max_results + 1 as a truncation probe must always stay inside that
# internal hard cap without the service needing to clamp it, and public
# responses stay bounded to a size actually meant for a caller, not an
# internal candidate pool.
MAX_PUBLIC_RESULTS = 25


class RetrievalStrategy(StrEnum):
    """Which retrieval capability answers a query. Closed on purpose.

    These are the application's supported retrieval strategies, not
    open-ended provider identities like ``DocumentSource.source_type``, so
    arbitrary strings are never accepted here; a CLI or MCP adapter is
    where a raw string would get parsed into one of these.
    """

    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"


class RetrievalStatus(StrEnum):
    """The shape of a valid retrieval outcome. Never a failure state.

    SUCCESS: one or more results, not known to be truncated.
    EMPTY: a valid query that matched nothing.
    PARTIAL: one or more results, and the public max_results bound is
    known to have truncated further ranked results.
    """

    SUCCESS = "success"
    EMPTY = "empty"
    PARTIAL = "partial"


class RetrievalHit(BaseModel):
    """One normalized public search result, from any strategy.

    The five strategy-specific fields below are legitimately optional
    here: this is the deliberately normalized cross-strategy public
    representation, not an attempt to force lexical/vector internals into
    one awkward shared model. A missing contribution is ``None``, never a
    fabricated ``0.0``. Native scores keep their own semantics rather than
    being renamed into a generic "relevance": ``bm25_score`` and
    ``vector_distance`` are both lower-is-better on their own strategy-
    specific scales, ``rrf_score`` is higher-is-better, and none of the
    three are comparable to each other.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: Chunk
    source_reference: SourceReference
    rank: int = Field(ge=1)

    lexical_rank: int | None = None
    vector_rank: int | None = None

    bm25_score: float | None = None
    vector_distance: float | None = None
    rrf_score: float | None = None


class RetrievalResult(BaseModel):
    """The public outcome of one search call.

    ``truncation_reason`` currently has exactly one possible value: the
    public max_results bound is the only source of truncation this service
    knows about. It stays a ``Literal`` rather than a free string so a
    caller can match on it without guessing what values exist.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: RetrievalStrategy
    status: RetrievalStatus
    results: tuple[RetrievalHit, ...]
    requested_max_results: int = Field(ge=1)
    truncated: bool
    truncation_reason: Literal["max_results"] | None = None

    @model_validator(mode="after")
    def _check_status_invariants(self) -> RetrievalResult:
        has_results = bool(self.results)
        match self.status:
            case RetrievalStatus.EMPTY:
                valid = not has_results and not self.truncated and self.truncation_reason is None
            case RetrievalStatus.SUCCESS:
                valid = has_results and not self.truncated and self.truncation_reason is None
            case RetrievalStatus.PARTIAL:
                valid = has_results and self.truncated and self.truncation_reason == "max_results"
        if not valid:
            raise ValueError(f"results/truncated/truncation_reason inconsistent with {self.status}")

        if len(self.results) > self.requested_max_results:
            raise ValueError("results must not exceed requested_max_results")

        expected_ranks = tuple(range(1, len(self.results) + 1))
        if tuple(hit.rank for hit in self.results) != expected_ranks:
            raise ValueError("result ranks must be 1-based and contiguous")

        return self


def _hit_from_lexical(match: LexicalMatch) -> RetrievalHit:
    return RetrievalHit(
        chunk=match.chunk,
        source_reference=match.source_reference,
        rank=match.rank,
        lexical_rank=match.rank,
        bm25_score=match.bm25_score,
    )


def _hit_from_vector(match: VectorMatch) -> RetrievalHit:
    return RetrievalHit(
        chunk=match.chunk,
        source_reference=match.source_reference,
        rank=match.rank,
        vector_rank=match.rank,
        vector_distance=match.distance,
    )


def _hit_from_hybrid(match: HybridMatch) -> RetrievalHit:
    return RetrievalHit(
        chunk=match.chunk,
        source_reference=match.source_reference,
        rank=match.rank,
        lexical_rank=match.lexical_rank,
        vector_rank=match.vector_rank,
        bm25_score=match.bm25_score,
        vector_distance=match.vector_distance,
        rrf_score=match.rrf_score,
    )


def _build_result(
    strategy: RetrievalStrategy, hits: list[RetrievalHit], max_results: int
) -> RetrievalResult:
    if not hits:
        return RetrievalResult(
            strategy=strategy,
            status=RetrievalStatus.EMPTY,
            results=(),
            requested_max_results=max_results,
            truncated=False,
        )

    # hits holds at most max_results + 1 entries (the truncation probe);
    # an (max_results + 1)-th entry is never shown, only used to know that
    # further ranked results exist beyond the public bound.
    if len(hits) > max_results:
        return RetrievalResult(
            strategy=strategy,
            status=RetrievalStatus.PARTIAL,
            results=tuple(hits[:max_results]),
            requested_max_results=max_results,
            truncated=True,
            truncation_reason="max_results",
        )

    return RetrievalResult(
        strategy=strategy,
        status=RetrievalStatus.SUCCESS,
        results=tuple(hits),
        requested_max_results=max_results,
        truncated=False,
    )


class KnowledgeRepository(Protocol):
    """The authoritative reads KnowledgeService actually consumes.

    Deliberately not ``persistence.base.Repository``: that Protocol also
    carries ``sync_source`` and the other write/listing methods ingestion
    needs, which this layer has no business depending on. Keeping this
    port here, owned by retrieval rather than persistence, means the
    retrieval package never has to import the persistence package at all;
    ``SqliteRepository`` already satisfies this structurally, with no
    inheritance and no adapter class required.
    """

    def get_document(self, document_id: str) -> Document | None: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...


class KnowledgeService:
    """Public retrieval boundary: strategy selection, no internals leaked.

    Constructing this with a ``vector_retriever`` also builds one
    ``HybridRetriever`` internally from ``lexical_index`` and that
    retriever; a caller never needs to assemble hybrid composition itself.
    Requesting VECTOR or HYBRID without a configured ``vector_retriever``
    raises ``VectorIndexUnavailableError`` outright, never a silent
    fallback to lexical.
    """

    def __init__(
        self,
        repository: KnowledgeRepository,
        lexical_index: LexicalIndex,
        vector_retriever: VectorRetriever | None = None,
        *,
        default_strategy: RetrievalStrategy = RetrievalStrategy.LEXICAL,
    ) -> None:
        self._repository = repository
        self._lexical_index = lexical_index
        self._vector_retriever = vector_retriever
        self._hybrid_retriever = (
            HybridRetriever(lexical_index, vector_retriever)
            if vector_retriever is not None
            else None
        )
        self._default_strategy = default_strategy

    def search(
        self,
        query: str,
        *,
        strategy: RetrievalStrategy | None = None,
        max_results: int = DEFAULT_PUBLIC_MAX_RESULTS,
    ) -> RetrievalResult:
        stripped = query.strip()
        if not stripped:
            raise InvalidQueryError("query must not be blank")
        if max_results <= 0:
            raise InvalidQueryError("max_results must be positive")
        if max_results > MAX_PUBLIC_RESULTS:
            raise InvalidQueryError(f"max_results must not exceed {MAX_PUBLIC_RESULTS}")

        effective_strategy = strategy if strategy is not None else self._default_strategy
        # One extra result is the entire truncation-detection mechanism:
        # asking for max_results + 1 and seeing whether that many come back
        # is a known-truncation signal, not a guess based on hitting the
        # requested count exactly.
        probe_limit = max_results + 1

        hits: list[RetrievalHit]
        if effective_strategy is RetrievalStrategy.LEXICAL:
            lexical_matches = self._lexical_index.search(stripped, max_results=probe_limit)
            hits = [_hit_from_lexical(match) for match in lexical_matches]
        elif effective_strategy is RetrievalStrategy.VECTOR:
            if self._vector_retriever is None:
                raise VectorIndexUnavailableError(
                    "vector retrieval requires a configured VectorRetriever"
                )
            vector_matches = self._vector_retriever.search(stripped, max_results=probe_limit)
            hits = [_hit_from_vector(match) for match in vector_matches]
        else:
            if self._hybrid_retriever is None:
                raise VectorIndexUnavailableError(
                    "hybrid retrieval requires a configured VectorRetriever"
                )
            hybrid_matches = self._hybrid_retriever.search(stripped, max_results=probe_limit)
            hits = [_hit_from_hybrid(match) for match in hybrid_matches]

        return _build_result(effective_strategy, hits, max_results)

    def get_document(self, document_id: str) -> Document | None:
        return self._repository.get_document(document_id)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._repository.get_chunk(chunk_id)
