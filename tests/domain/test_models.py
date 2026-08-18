from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from engineering_knowledge.domain.identity import derive_chunk_id, hash_content
from engineering_knowledge.domain.models import (
    Chunk,
    Document,
    SectionPath,
    SourceReference,
)


def test_section_path_canonicalization() -> None:
    path = SectionPath(headings=("  Deployment  ", "Rollback Procedure"))
    assert path.headings == ("Deployment", "Rollback Procedure")

    with pytest.raises(ValidationError):
        SectionPath(headings=("Deployment", "   "))


def test_document_identity_stable_under_equivalent_paths_and_content_changes() -> None:
    ingested_at = datetime.now(UTC)
    posix = Document.create(
        source_id="runbooks",
        relative_path="deploy/aegis.md",
        title="Aegis deploy guide",
        content_hash=hash_content("version one"),
        ingested_at=ingested_at,
    )
    windows_style_same_path = Document.create(
        source_id="runbooks",
        relative_path="deploy\\aegis.md",
        title="Aegis deploy guide",
        content_hash=hash_content("version two"),
        ingested_at=ingested_at,
    )

    assert posix.document_id == windows_style_same_path.document_id
    assert posix.content_hash != windows_style_same_path.content_hash


def test_document_ingested_at_normalizes_to_utc_and_requires_tzinfo() -> None:
    eastern = timezone(timedelta(hours=-5))
    local_time = datetime(2026, 1, 1, 9, 0, tzinfo=eastern)
    document = Document.create(
        source_id="runbooks",
        relative_path="deploy/aegis.md",
        title="Aegis deploy guide",
        content_hash=hash_content("v1"),
        ingested_at=local_time,
    )
    assert document.ingested_at == local_time
    assert document.ingested_at.tzinfo == UTC

    with pytest.raises(ValidationError):
        Document.create(
            source_id="runbooks",
            relative_path="deploy/aegis.md",
            title="Aegis deploy guide",
            content_hash=hash_content("v1"),
            ingested_at=datetime.now(),
        )


def test_chunk_identity_stable_under_text_and_unrelated_ordinal_changes() -> None:
    section = SectionPath(headings=("Deployment", "Rollback"))

    original = Chunk.create(
        document_id="doc_abc", section_path=section, section_occurrence=0,
        ordinal=1, ordinal_in_section=0, text="run script A",
    )
    edited_text = Chunk.create(
        document_id="doc_abc", section_path=section, section_occurrence=0,
        ordinal=1, ordinal_in_section=0, text="run script A, updated",
    )
    shifted_by_earlier_section = Chunk.create(
        document_id="doc_abc", section_path=section, section_occurrence=0,
        ordinal=6, ordinal_in_section=0, text="run script A",
    )
    second_occurrence = Chunk.create(
        document_id="doc_abc", section_path=section, section_occurrence=1,
        ordinal=1, ordinal_in_section=0, text="run script A",
    )

    assert original.chunk_id == edited_text.chunk_id
    assert original.content_hash != edited_text.content_hash
    assert original.chunk_id == shifted_by_earlier_section.chunk_id
    assert original.chunk_id != second_occurrence.chunk_id


def test_direct_construction_rejects_inconsistent_derived_identity() -> None:
    with pytest.raises(ValidationError):
        Document(
            document_id="doc_000000000000000000000000",
            source_id="runbooks",
            relative_path="deploy/aegis.md",
            title="Aegis deploy guide",
            content_hash=hash_content("v1"),
            ingested_at=datetime.now(UTC),
        )

    section = SectionPath(headings=("Deployment",))
    text = "rollback steps"
    valid_chunk_id = derive_chunk_id("doc_abc", section.as_identity_string(), 0, 0)

    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="chunk_000000000000000000000000",
            document_id="doc_abc",
            section_path=section,
            section_occurrence=0,
            ordinal=0,
            ordinal_in_section=0,
            text=text,
            content_hash=hash_content(text),
            char_count=len(text),
        )

    with pytest.raises(ValidationError):
        Chunk(
            chunk_id=valid_chunk_id,
            document_id="doc_abc",
            section_path=section,
            section_occurrence=0,
            ordinal=0,
            ordinal_in_section=0,
            text=text,
            content_hash=hash_content("mismatched content"),
            char_count=len(text),
        )


def test_source_reference_preserves_provenance_and_rejects_document_mismatch() -> None:
    document = Document.create(
        source_id="runbooks",
        relative_path="deploy/aegis.md",
        title="Aegis deploy guide",
        content_hash=hash_content("v1"),
        ingested_at=datetime.now(UTC),
    )
    section = SectionPath(headings=("Deployment", "Rollback"))
    chunk = Chunk.create(
        document_id=document.document_id,
        section_path=section,
        section_occurrence=1,
        ordinal=0,
        ordinal_in_section=0,
        text="rollback steps",
    )

    reference = SourceReference.from_chunk(chunk=chunk, document=document)

    assert reference.source_id == document.source_id
    assert reference.document_id == document.document_id
    assert reference.chunk_id == chunk.chunk_id
    assert reference.relative_path == document.relative_path
    assert reference.section_path == section
    assert reference.section_occurrence == 1
    assert reference.content_hash == chunk.content_hash

    unrelated_chunk = Chunk.create(
        document_id="doc_unrelated",
        section_path=section,
        section_occurrence=0,
        ordinal=0,
        ordinal_in_section=0,
        text="rollback steps",
    )
    with pytest.raises(ValueError):
        SourceReference.from_chunk(chunk=unrelated_chunk, document=document)
