"""Vector retrieval boundary: the port hybrid retrieval will later consume.

Mirrors ``retrieval.lexical``'s shape deliberately. ``VectorIndex`` is one
method, nearest-neighbor search against a query vector; index maintenance
is a persistence-adapter concern, not part of this port. ``VectorRetriever``
is the thin composition layer that lets a caller search with ordinary query
text instead of a vector: it embeds the query, then asks the index, and
nothing more. It never calls lexical search, never fuses scores, never
reranks; that is later work, not this port's job.

Distance is FTS5's counterpart here: a vector engine's native similarity
score, strategy-specific and not comparable across retrieval strategies.
Hybrid retrieval will fuse ranks, not raw distances or BM25 scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from engineering_knowledge.domain import Chunk, SourceReference
from engineering_knowledge.embeddings.base import EmbeddingProvider
from engineering_knowledge.embeddings.fingerprint import derive_embedding_fingerprint
from engineering_knowledge.retrieval.errors import InvalidQueryError
from engineering_knowledge.retrieval.lexical import DEFAULT_MAX_RESULTS, MAX_RESULTS


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """One ranked vector result, with structured provenance already attached.

    ``rank`` is 1-based. ``distance`` is the vector index's native nearest-
    neighbor score (lower means more similar for the engine this port is
    built against); it has no meaning outside this retrieval strategy and is
    never normalized into a fake universal relevance value.
    """

    chunk: Chunk
    source_reference: SourceReference
    rank: int
    distance: float


class VectorIndex(Protocol):
    """Searches persisted chunk vectors for nearest neighbors of a query vector.

    ``embedding_fingerprint`` is required, not optional: the index uses it
    to confirm the query vector was produced by the same provider/model
    that produced the persisted vectors, and raises
    ``VectorIndexIncompatibleError`` rather than silently comparing vectors
    from different embedding spaces.
    """

    def search_vector(
        self,
        query_vector: tuple[float, ...],
        *,
        embedding_fingerprint: str,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> tuple[VectorMatch, ...]: ...


class VectorRetriever:
    """Composes an EmbeddingProvider with a VectorIndex to search by query text.

    The only responsibilities here are validating the query, embedding it,
    and delegating to the index. No hybrid logic, no lexical calls, no
    answer generation, no reranking.
    """

    def __init__(self, embedding_provider: EmbeddingProvider, vector_index: VectorIndex) -> None:
        self._embedding_provider = embedding_provider
        self._vector_index = vector_index

    def search(
        self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS
    ) -> tuple[VectorMatch, ...]:
        stripped = query.strip()
        if not stripped:
            raise InvalidQueryError("query must not be blank")
        if max_results <= 0:
            raise InvalidQueryError("max_results must be positive")
        if max_results > MAX_RESULTS:
            raise InvalidQueryError(f"max_results must not exceed {MAX_RESULTS}")

        provider = self._embedding_provider
        query_vector = provider.embed_query(stripped)
        embedding_fingerprint = derive_embedding_fingerprint(
            provider_type=provider.provider_type,
            model_id=provider.model_id,
            model_revision=provider.model_revision,
            dimension=provider.dimension,
        )

        return self._vector_index.search_vector(
            query_vector, embedding_fingerprint=embedding_fingerprint, max_results=max_results
        )
