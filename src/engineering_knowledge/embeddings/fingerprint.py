"""Deterministic embedding fingerprinting and embedding-input text.

``embedding_fingerprint`` answers a different question than
``ingestion.processing.processing_fingerprint``: whether the searchable
representation a chunk was embedded from, and the provider/model that
embedded it, are what a persisted vector still reflects. Changing the
embedding provider or model must invalidate vector state without touching
chunks, the same way changing ``max_chunk_chars`` invalidates chunks
without touching source content. The two fingerprints are computed the
same way (canonical JSON, full SHA-256) but never mixed together.
"""

from __future__ import annotations

import hashlib
import json

from engineering_knowledge.domain import SectionPath

# Bumping this invalidates every persisted embedding_fingerprint, forcing a
# vector rebuild, even if the provider/model is unchanged: it exists for
# when this composition itself changes, independent of the model.
EMBEDDING_TEXT_VERSION = "title-section-content-v1"


def build_embedding_text(*, title: str, section_path: SectionPath, chunk_text: str) -> str:
    """Deterministic text representation embedded for semantic retrieval.

    Composes title, section heading hierarchy, and chunk text, in that
    order. The root section contributes no heading line rather than a
    placeholder. Never touches ``Chunk.text``; this exists only as
    embedding input, not as anything persisted onto the chunk itself.
    """
    parts = [title]
    if section_path.headings:
        parts.append(" ".join(section_path.headings))
    parts.append(chunk_text)
    return "\n".join(parts)


def derive_embedding_fingerprint(
    *, provider_type: str, model_id: str, model_revision: str | None, dimension: int
) -> str:
    """Fingerprint the provider/model/representation that produces vectors.

    Deliberately excludes document content, source_id, relative_path, API
    keys, and timestamps: those describe what was embedded, not what would
    produce the same vector space again.
    """
    payload = {
        "provider_type": provider_type,
        "model_id": model_id,
        "model_revision": model_revision,
        "dimension": dimension,
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
