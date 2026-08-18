"""Turns one RawDocument into a ProcessedDocument, deterministically.

Two independent questions decide whether a previously persisted document
needs work: did its normalized content change, and did the algorithm or
configuration that turns that content into chunks change. content_hash
answers the first. processing_fingerprint answers the second, and exists
because content_hash alone cannot: a document whose source bytes never
changed can still need reprocessing after we lower max_chunk_chars, change
how sections are parsed, or otherwise ship a chunking behavior change.
Without a separate fingerprint, that kind of change would leave stale
chunks persisted forever alongside an unchanged content_hash.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from engineering_knowledge.chunking import chunk_markdown, chunk_plain_text, normalize_text
from engineering_knowledge.domain import Document
from engineering_knowledge.domain.identity import derive_document_id, hash_content
from engineering_knowledge.persistence.base import ProcessedDocument
from engineering_knowledge.sources.base import RawDocument

# Bumping either version invalidates every previously computed
# processing_fingerprint, forcing reprocessing, even if nothing else here
# changes. This is independent of the package release version: 0.1.0 is a
# distribution concern, not a statement about what algorithm produced a
# given persisted chunk set.
NORMALIZATION_VERSION = "newline-v1"
CHUNKING_VERSION = "section-aware-v1"

DocumentFormat = Literal["markdown", "plain_text"]

_EXTENSION_FORMATS: dict[str, DocumentFormat] = {
    ".md": "markdown",
    ".txt": "plain_text",
}


class IngestionError(Exception):
    """Base class for expected ingestion-boundary failures."""


class UnsupportedDocumentFormatError(IngestionError):
    """A discovered document's extension has no known processing format."""


def determine_format(relative_path: str) -> DocumentFormat:
    """Map a relative path's extension to a processing format.

    Matching is case-insensitive; the path itself is never rewritten. An
    extension outside {.md, .txt} is refused explicitly rather than
    guessed at: a source adapter's own extension allowlist is a separate,
    independently configured concern from what this layer knows how to
    process, and silently treating an unknown format as plain text would
    hide a real configuration mismatch.
    """
    suffix = PurePosixPath(relative_path).suffix.lower()
    document_format = _EXTENSION_FORMATS.get(suffix)
    if document_format is None:
        raise UnsupportedDocumentFormatError(
            f"no processing format for extension {suffix!r}: {relative_path!r}"
        )
    return document_format


def derive_processing_fingerprint(*, document_format: DocumentFormat, max_chunk_chars: int) -> str:
    """Fingerprint the processing behavior that would turn content into chunks.

    Deliberately excludes document content, source_id, relative_path, and
    any timestamp: those describe what was processed, not how. This is a
    content-addressing hash of processing behavior, not a short id, so it
    keeps the full SHA-256 hex digest.
    """
    payload = {
        "normalization_version": NORMALIZATION_VERSION,
        "chunking_version": CHUNKING_VERSION,
        "document_format": document_format,
        "max_chunk_chars": max_chunk_chars,
        # No overlap between chunks exists in v0.1.0; recorded explicitly
        # so that adding overlap later is a visible fingerprint change,
        # not an implicit one.
        "overlap": None,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def process_raw_document(
    raw: RawDocument, *, max_chunk_chars: int, now: datetime
) -> ProcessedDocument:
    """Normalize, identify, and chunk one RawDocument.

    ``now`` is the ingestion attempt's timestamp, used as this document's
    candidate ingested_at. Whether it actually gets persisted, or the
    previous ingested_at is preserved instead, is decided by the
    repository during sync, not here: this function has no notion of what
    was previously persisted.
    """
    document_format = determine_format(raw.relative_path)
    normalized = normalize_text(raw.content)
    content_hash = hash_content(normalized)
    document_id = derive_document_id(raw.source_id, raw.relative_path)
    processing_fingerprint = derive_processing_fingerprint(
        document_format=document_format, max_chunk_chars=max_chunk_chars
    )
    title = PurePosixPath(raw.relative_path).name

    if document_format == "markdown":
        chunks = chunk_markdown(document_id, normalized, max_chunk_chars=max_chunk_chars)
    else:
        chunks = chunk_plain_text(document_id, normalized, max_chunk_chars=max_chunk_chars)

    document = Document.create(
        source_id=raw.source_id,
        relative_path=raw.relative_path,
        title=title,
        content_hash=content_hash,
        ingested_at=now,
    )
    return ProcessedDocument(
        document=document, chunks=chunks, processing_fingerprint=processing_fingerprint
    )
