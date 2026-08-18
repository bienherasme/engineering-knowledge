"""Normalized domain models for engineering knowledge.

These models represent knowledge after it has been identified and located,
not while it is being read from disk or embedded. Nothing here knows about
the filesystem, SQLite, FTS5, embeddings, or MCP; those belong to the
adapters that build these models, never the other way around.

Models are frozen and reject unexpected fields, so an instance in hand is
exactly what it claims to be. Identity and content-consistency fields
(``document_id``, ``chunk_id``, ``char_count``) are validated against the
same deterministic rules regardless of whether a model was built through its
``create`` factory or constructed directly, for example when reconstructing
one from persisted storage.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from engineering_knowledge.domain.identity import (
    derive_chunk_id,
    derive_document_id,
    hash_content,
    normalize_relative_path,
)

_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _require_sha256_hex(value: str) -> str:
    if not _SHA256_HEX_PATTERN.fullmatch(value):
        raise ValueError("must be a lowercase sha256 hex digest")
    return value


class DocumentSource(BaseModel):
    """A configured knowledge source.

    ``source_id`` is a stable key assigned by configuration, not derived
    from filesystem layout. It deliberately excludes a root path or any
    other adapter-specific configuration; that detail belongs to the source
    adapter itself, never to this identity model.

    ``source_type`` is a free-form, validated string rather than a closed
    enum, so a future adapter (Confluence, Drive, an internal wiki) never
    requires editing this model to be expressible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    source_type: str
    display_name: str | None = None

    @field_validator("source_id", "source_type")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        return _require_non_blank(value)

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("display_name must not be blank when provided")
        return value


class SectionPath(BaseModel):
    """Ordered heading hierarchy locating a chunk within a document.

    Leading and trailing whitespace around a heading is not meaningful and
    is stripped, but case and internal spacing are treated as part of the
    heading's identity and are preserved as-is. An empty tuple represents
    content that precedes any heading.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    headings: tuple[str, ...] = ()

    @field_validator("headings", mode="before")
    @classmethod
    def _normalize_headings(cls, value: Sequence[str]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for heading in value:
            stripped = heading.strip()
            if not stripped:
                raise ValueError("section heading must not be blank")
            cleaned.append(stripped)
        return tuple(cleaned)

    def as_identity_string(self) -> str:
        """Canonical form used as an identity input, not for display."""
        return "\x1f".join(self.headings)


class Document(BaseModel):
    """A single ingested document, identified independently of its content.

    ``document_id`` is derived from ``(source_id, relative_path)`` and is
    verified against that derivation on every construction. The same
    logical document keeps the same id even after its content changes;
    ``content_hash`` is what identifies the version that was actually
    ingested. ``content_hash`` is supplied by the caller rather than
    computed here, because this model does not retain the document's full
    text, only its identity and provenance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    source_id: str
    relative_path: str
    title: str
    content_hash: str
    ingested_at: datetime
    document_type: str | None = None

    @field_validator("source_id", "title")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        return _require_non_blank(value)

    @field_validator("document_type")
    @classmethod
    def _validate_document_type(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("document_type must not be blank when provided")
        return value

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        normalized = normalize_relative_path(value)
        if normalized != value:
            raise ValueError(
                f"relative_path must already be normalized: {value!r} != {normalized!r}"
            )
        return value

    @field_validator("content_hash")
    @classmethod
    def _validate_content_hash(cls, value: str) -> str:
        return _require_sha256_hex(value)

    @field_validator("ingested_at")
    @classmethod
    def _validate_ingested_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("ingested_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _check_document_id(self) -> Document:
        expected = derive_document_id(self.source_id, self.relative_path)
        if self.document_id != expected:
            raise ValueError("document_id does not match its derived identity")
        return self

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        relative_path: str,
        title: str,
        content_hash: str,
        ingested_at: datetime,
        document_type: str | None = None,
    ) -> Document:
        normalized_path = normalize_relative_path(relative_path)
        document_id = derive_document_id(source_id, normalized_path)
        return cls(
            document_id=document_id,
            source_id=source_id,
            relative_path=normalized_path,
            title=title,
            content_hash=content_hash,
            ingested_at=ingested_at,
            document_type=document_type,
        )


class Chunk(BaseModel):
    """A logically identified slice of a document's content.

    ``chunk_id`` is derived from ``(document_id, section_path,
    section_occurrence, ordinal_in_section)`` and deliberately excludes both
    content and the global ``ordinal``, so editing this chunk's text, or
    adding chunks elsewhere in the document, never changes this chunk's
    identity. ``section_occurrence`` distinguishes two sections that share
    the same human-visible heading path (two "## Examples" sections in the
    same document, for instance); without it their chunks would collide on
    identity. ``content_hash`` and ``char_count`` are verified against
    ``text`` on every construction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    document_id: str
    section_path: SectionPath
    section_occurrence: int = Field(ge=0)
    ordinal: int = Field(ge=0)
    ordinal_in_section: int = Field(ge=0)
    text: str = Field(min_length=1)
    content_hash: str
    char_count: int = Field(ge=0)

    @field_validator("document_id")
    @classmethod
    def _validate_document_id(cls, value: str) -> str:
        return _require_non_blank(value)

    @field_validator("content_hash")
    @classmethod
    def _validate_content_hash(cls, value: str) -> str:
        return _require_sha256_hex(value)

    @model_validator(mode="after")
    def _check_identity_and_content(self) -> Chunk:
        expected_chunk_id = derive_chunk_id(
            self.document_id,
            self.section_path.as_identity_string(),
            self.section_occurrence,
            self.ordinal_in_section,
        )
        if self.chunk_id != expected_chunk_id:
            raise ValueError("chunk_id does not match its derived identity")
        if self.char_count != len(self.text):
            raise ValueError("char_count must equal len(text)")
        if self.content_hash != hash_content(self.text):
            raise ValueError("content_hash must equal hash_content(text)")
        return self

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        section_path: SectionPath,
        section_occurrence: int,
        ordinal: int,
        ordinal_in_section: int,
        text: str,
    ) -> Chunk:
        chunk_id = derive_chunk_id(
            document_id,
            section_path.as_identity_string(),
            section_occurrence,
            ordinal_in_section,
        )
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            section_path=section_path,
            section_occurrence=section_occurrence,
            ordinal=ordinal,
            ordinal_in_section=ordinal_in_section,
            text=text,
            content_hash=hash_content(text),
            char_count=len(text),
        )


class SourceReference(BaseModel):
    """Structured provenance making a retrieved chunk independently traceable.

    Every field here answers one specific provenance question (which
    source, which document, which chunk, which section, which content
    version), rather than folding provenance into an opaque metadata dict
    or a preformatted citation string. A human-readable citation can be
    assembled from these fields later, by whichever layer needs one.

    ``section_occurrence`` is included alongside ``section_path`` because
    provenance must be able to distinguish two duplicate human-visible
    section paths within the same document, not just name the path they
    share.

    ``ingested_at`` is deliberately not included: ``content_hash`` already
    identifies the observed version, and the ingestion timestamp belongs to
    ``Document`` rather than to each individual reference into it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    document_id: str
    chunk_id: str
    relative_path: str
    section_path: SectionPath
    section_occurrence: int = Field(ge=0)
    content_hash: str

    @classmethod
    def from_chunk(cls, *, chunk: Chunk, document: Document) -> SourceReference:
        if chunk.document_id != document.document_id:
            raise ValueError("chunk does not belong to document")
        return cls(
            source_id=document.source_id,
            document_id=document.document_id,
            chunk_id=chunk.chunk_id,
            relative_path=document.relative_path,
            section_path=chunk.section_path,
            section_occurrence=chunk.section_occurrence,
            content_hash=chunk.content_hash,
        )
