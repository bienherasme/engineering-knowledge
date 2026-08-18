"""Embedding provider boundary.

``EmbeddingProvider`` is a small provider-neutral port: given text, return a
vector. It knows nothing about storage, indexing, or retrieval ranking, and
nothing here imports sqlite or a specific model library. Query and document
embedding are kept as separate methods rather than one shared call, because
some models compute genuinely different representations for a search query
than for the passage it might match; a provider that treats them identically
is free to implement both methods the same way, but the port does not force
that assumption on providers that cannot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol


class EmbeddingError(Exception):
    """A provider produced invalid output: wrong dimension or a non-finite value."""


class EmbeddingConfigurationError(EmbeddingError):
    """A provider or its underlying model could not be constructed or configured."""


class EmbeddingProvider(Protocol):
    """Turns text into fixed-dimension vectors for semantic retrieval.

    ``provider_type``, ``model_id``, ``model_revision``, and ``dimension``
    are identity metadata, not behavior: they exist so a caller can derive
    an ``embedding_fingerprint`` (see ``embeddings.fingerprint``) without
    needing to know anything about how a specific provider works.
    """

    provider_type: str
    model_id: str
    model_revision: str | None
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


def validate_embedding_vector(
    vector: Sequence[float], *, expected_dimension: int
) -> tuple[float, ...]:
    """Validate and normalize one embedding vector into the port's representation.

    Centralized here so every provider enforces the same two invariants
    before a vector leaves the embedding boundary: it must match the
    provider's declared dimension, and every value must be finite. A
    dimension mismatch or a NaN/infinite value is a provider bug, not
    something a caller downstream should have to detect.
    """
    if len(vector) != expected_dimension:
        raise EmbeddingError(
            f"embedding vector has dimension {len(vector)}, expected {expected_dimension}"
        )
    values = tuple(float(value) for value in vector)
    for value in values:
        if not math.isfinite(value):
            raise EmbeddingError(f"embedding vector contains a non-finite value: {value}")
    return values
