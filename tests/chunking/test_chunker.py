import pytest

from engineering_knowledge.chunking import (
    ChunkingError,
    chunk_markdown,
    chunk_plain_text,
    normalize_text,
)
from engineering_knowledge.domain import SectionPath


def test_normalize_text_normalizes_line_endings_and_is_idempotent() -> None:
    raw = "line one\r\nline two\rline three\n"
    normalized = normalize_text(raw)
    assert normalized == "line one\nline two\nline three\n"
    assert normalize_text(normalized) == normalized


def test_chunk_markdown_builds_expected_section_hierarchy() -> None:
    doc = (
        "##\n"
        "# Payments\n"
        "intro\n"
        "#\n"
        "## Deployment\n"
        "deploy notes\n"
        "### Rollback\n"
        "rollback notes\n"
        "##### Deep Skip\n"
        "deep notes\n"
        "# Billing\n"
        "billing notes\n"
        "### ###\n"
    )
    chunks = chunk_markdown("doc_hierarchy", doc, max_chunk_chars=1000)
    assert [chunk.section_path.headings for chunk in chunks] == [
        (),
        ("Payments",),
        ("Payments", "Deployment"),
        ("Payments", "Deployment", "Rollback"),
        # levels 2 and 4 were never present here, and no placeholder for
        # them is invented: the path reflects only headings actually seen.
        ("Payments", "Deployment", "Rollback", "Deep Skip"),
        ("Billing",),
    ]

    # "##", "#", and "### ###" all look structurally like ATX headings but
    # their canonical title strips to nothing, so none of them introduce a
    # section: each stays as ordinary content of whatever section is
    # already open (root, for the first one), preserved verbatim.
    assert chunks[0].text == "##\n"
    assert "#\n" in chunks[1].text
    assert chunks[-1].text.endswith("### ###\n")

    reconstructed = "".join(chunk.text for chunk in chunks)
    assert reconstructed == normalize_text(doc)


def test_chunk_markdown_ignores_headings_inside_fenced_code_block() -> None:
    doc = "# Payments\n```python\n# not a heading\n```\n## Deployment\nnotes\n"
    chunks = chunk_markdown("doc_fence", doc, max_chunk_chars=1000)
    assert [chunk.section_path.headings for chunk in chunks] == [
        ("Payments",),
        ("Payments", "Deployment"),
    ]
    assert "# not a heading" in chunks[0].text


def test_chunk_markdown_preamble_and_chunk_plain_text_use_root_section() -> None:
    doc = "preamble line\nmore preamble\n# First Heading\nbody\n"
    chunks = chunk_markdown("doc_preamble", doc, max_chunk_chars=1000)
    assert chunks[0].section_path == SectionPath()
    assert chunks[0].text == "preamble line\nmore preamble\n"

    plain_chunks = chunk_plain_text(
        "doc_plain", "just plain text\nsecond line\n", max_chunk_chars=1000
    )
    assert len(plain_chunks) == 1
    assert plain_chunks[0].section_path == SectionPath()

    assert chunk_markdown("doc_empty", "", max_chunk_chars=1000) == ()


def test_chunk_markdown_repeated_section_path_gets_distinct_occurrence_and_chunk_id() -> None:
    doc = "## Examples\nfirst\n## Examples\nsecond\n"
    chunks = chunk_markdown("doc_repeat", doc, max_chunk_chars=1000)

    assert len(chunks) == 2
    assert chunks[0].section_path.headings == ("Examples",)
    assert chunks[1].section_path.headings == ("Examples",)
    assert chunks[0].section_occurrence == 0
    assert chunks[1].section_occurrence == 1
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_chunk_id_stable_under_unrelated_changes() -> None:
    base = "## Deployment\nrun script A\n"
    edited_text = "## Deployment\nrun script A, updated\n"
    with_earlier_section = "## Unrelated\nsomething else entirely\n## Deployment\nrun script A\n"

    base_chunk = chunk_markdown("doc_stable", base, max_chunk_chars=1000)[0]
    edited_chunk = chunk_markdown("doc_stable", edited_text, max_chunk_chars=1000)[0]
    shifted_chunk = chunk_markdown("doc_stable", with_earlier_section, max_chunk_chars=1000)[-1]

    assert base_chunk.chunk_id == edited_chunk.chunk_id
    assert base_chunk.content_hash != edited_chunk.content_hash
    assert base_chunk.chunk_id == shifted_chunk.chunk_id


def test_chunk_splitting_is_deterministic_and_respects_bound() -> None:
    doc = "## Section\n" + ("word " * 5) + "\n" + ("x" * 37) + "\n" + ("more words " * 4) + "\n"

    first = chunk_markdown("doc_bound", doc, max_chunk_chars=15)
    second = chunk_markdown("doc_bound", doc, max_chunk_chars=15)

    assert first == second
    assert all(len(chunk.text) <= 15 for chunk in first)


def test_chunk_reconstruction_matches_normalized_source() -> None:
    doc = (
        "preamble\n\n"
        "# Payments\n\n"
        "Some intro text that is reasonably long for a paragraph.\n\n"
        "## Deployment\n\n"
        "Step one.\nStep two.\n\n"
        "Step three after a blank line.\n"
    )
    chunks = chunk_markdown("doc_reconstruct", doc, max_chunk_chars=40)
    reconstructed = "".join(chunk.text for chunk in chunks)
    assert reconstructed == normalize_text(doc)


@pytest.mark.parametrize("max_chunk_chars", [0, -5])
def test_chunk_markdown_rejects_non_positive_max_chunk_chars(max_chunk_chars: int) -> None:
    with pytest.raises(ChunkingError):
        chunk_markdown("doc_invalid", "text", max_chunk_chars=max_chunk_chars)
