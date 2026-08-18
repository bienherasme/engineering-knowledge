"""Deterministic identity helpers for the domain layer.

Every identifier here is derived from stable inputs only: no random UUIDs, no
timestamps, no filesystem state, no locale-dependent behavior. The same
inputs always produce the same identifier, on any machine, on any run.

IDs are namespaced digests (``doc_<hex>``, ``chunk_<hex>``) truncated to
``_ID_DIGEST_LENGTH`` hex characters. At 24 hex characters that is 96 bits of
SHA-256, which is practically collision-safe at the scale this project
operates at and keeps IDs short enough to be useful in logs and citations.

Content hashing (``hash_content``) is unrelated to identity and uses the
full SHA-256 hex digest, since its job is to detect any change in content,
not to name something.
"""

from __future__ import annotations

import hashlib
import re

_ID_DIGEST_LENGTH = 24

_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


def normalize_relative_path(path: str) -> str:
    """Normalize a source-relative path into a stable, portable identity input.

    Accepts POSIX or Windows-style separators so that identity does not
    depend on which machine performed the ingestion. Absolute paths, parent
    traversal, and empty segments are identity hazards and are rejected
    rather than silently repaired; only semantically unnecessary ``.``
    segments are dropped.
    """
    if not path.strip():
        raise ValueError("relative_path must not be empty")

    slash_path = path.replace("\\", "/")

    if slash_path.startswith("/"):
        raise ValueError(f"relative_path must not be absolute: {path!r}")
    if _WINDOWS_DRIVE_PATTERN.match(slash_path):
        raise ValueError(f"relative_path must not be absolute: {path!r}")

    segments: list[str] = []
    for segment in slash_path.split("/"):
        if segment == ".":
            continue
        if segment == "":
            raise ValueError(f"relative_path must not contain empty segments: {path!r}")
        if segment == "..":
            raise ValueError(f"relative_path must not contain parent traversal: {path!r}")
        segments.append(segment)

    if not segments:
        raise ValueError(f"relative_path must resolve to at least one segment: {path!r}")

    return "/".join(segments)


def hash_content(text: str) -> str:
    """Return the SHA-256 hex digest of exactly the text supplied.

    No normalization happens here. Whitespace collapsing, encoding cleanup,
    or any other text normalization is a concern of the future ingestion
    pipeline, not of this hashing primitive.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_document_id(source_id: str, relative_path: str) -> str:
    """Derive a stable logical document identifier.

    Identity is ``(source_id, normalized relative_path)`` only. Content and
    ingestion time never participate, so the same logical document keeps the
    same id across every re-ingestion no matter how its content changes.

    ``relative_path`` is normalized internally through
    ``normalize_relative_path`` before hashing, so this function is safe to
    call with any valid spelling of the same path: callers never need to
    know about a hidden canonicalization precondition. Two spellings that
    normalize to the same path always produce the same document id.
    """
    if not source_id.strip():
        raise ValueError("source_id must not be empty")

    normalized_path = normalize_relative_path(relative_path)
    digest_input = f"{source_id}\x1f{normalized_path}".encode()
    digest = hashlib.sha256(digest_input).hexdigest()[:_ID_DIGEST_LENGTH]
    return f"doc_{digest}"


def derive_chunk_id(
    document_id: str, section_identity: str, section_occurrence: int, ordinal_in_section: int
) -> str:
    """Derive a stable logical chunk identifier.

    Identity is ``(document_id, section_identity, section_occurrence,
    ordinal_in_section)``. ``section_occurrence`` exists because a document
    can contain two sections with the same human-visible heading path (two
    "## Examples" sections, for instance); without it, their chunks would
    collide on identity. It counts occurrences of the same canonical
    section_identity in document order, starting at 0, and is independent
    of any other section's occurrence count.

    Content never participates: editing a chunk's text changes its
    content_hash but keeps its chunk_id stable. A global ordinal is
    deliberately excluded from identity so that adding or removing chunks in
    an unrelated section never shifts this chunk's id.
    """
    if not document_id.strip():
        raise ValueError("document_id must not be empty")
    if section_occurrence < 0:
        raise ValueError("section_occurrence must not be negative")
    if ordinal_in_section < 0:
        raise ValueError("ordinal_in_section must not be negative")

    digest_input = (
        f"{document_id}\x1f{section_identity}\x1f{section_occurrence}\x1f{ordinal_in_section}"
    ).encode()
    digest = hashlib.sha256(digest_input).hexdigest()[:_ID_DIGEST_LENGTH]
    return f"chunk_{digest}"
