"""Deterministic, explicitly non-semantic embedding provider.

Exists only for plumbing: index synchronization, dimension validation,
deterministic nearest-neighbor wiring, and failure semantics. Never use this
to argue anything about vector or hybrid retrieval quality. Its vectors are
derived from a SHA-256 digest of the input text, not from any semantic
model, so two lexically different but semantically related texts produce
unrelated vectors, and there is no natural-language understanding of any
kind here, deliberately.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from engineering_knowledge.embeddings.base import validate_embedding_vector

DEFAULT_DIMENSION = 8


class FakeEmbeddingProvider:
    """Non-semantic, digest-derived embedding provider for tests and plumbing."""

    provider_type = "fake"
    model_id = "fake-digest-v1"
    model_revision: str | None = None

    def __init__(self, *, dimension: int = DEFAULT_DIMENSION) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed(text)

    def _embed(self, text: str) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw_bytes = [digest[i % len(digest)] for i in range(self.dimension)]
        # Scaled into a small, bounded, finite float range. This is
        # deterministic numeric plumbing, not a semantic embedding: the
        # scaling exists only so downstream code has ordinary-looking
        # floats to validate and store, nothing more.
        vector = tuple((value / 255.0) * 2.0 - 1.0 for value in raw_bytes)
        return validate_embedding_vector(vector, expected_dimension=self.dimension)
