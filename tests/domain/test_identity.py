import pytest

from engineering_knowledge.domain.identity import (
    derive_chunk_id,
    derive_document_id,
    normalize_relative_path,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("docs/runbook.md", "docs/runbook.md"),
        ("docs\\runbook.md", "docs/runbook.md"),
        ("docs/./runbook.md", "docs/runbook.md"),
    ],
)
def test_normalize_relative_path_equivalences(raw: str, expected: str) -> None:
    assert normalize_relative_path(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "/etc/passwd",
        "C:\\Users\\bob\\notes.md",
        "docs/../secrets.md",
    ],
)
def test_normalize_relative_path_rejects_invalid_input(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_relative_path(raw)


def test_derive_document_id_normalizes_input_before_hashing() -> None:
    # The public helper must not require callers to pre-canonicalize their
    # path: two equivalent spellings of the same relative path have to
    # produce the same document id without any hidden precondition.
    assert derive_document_id("docs", "runbooks/./payments.md") == derive_document_id(
        "docs", "runbooks\\payments.md"
    )


def test_derive_chunk_id_is_deterministic_and_excludes_ordinal() -> None:
    document_id = derive_document_id("runbooks", "deploy/aegis.md")

    baseline = derive_chunk_id(document_id, "Deployment\x1fRollback", 0, 0)
    repeat = derive_chunk_id(document_id, "Deployment\x1fRollback", 0, 0)
    different_section = derive_chunk_id(document_id, "Deployment\x1fVerification", 0, 0)
    different_occurrence = derive_chunk_id(document_id, "Deployment\x1fRollback", 1, 0)
    different_local_ordinal = derive_chunk_id(document_id, "Deployment\x1fRollback", 0, 1)

    assert baseline == repeat
    assert baseline != different_section
    assert baseline != different_occurrence
    assert baseline != different_local_ordinal
