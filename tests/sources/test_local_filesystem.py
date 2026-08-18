from pathlib import Path

import pytest

from engineering_knowledge.domain.identity import derive_document_id
from engineering_knowledge.sources import (
    LocalFilesystemSourceAdapter,
    SourceConfigurationError,
    SourceFileTooLargeError,
    SourceReadError,
)


def test_discover_yields_supported_files_recursively_in_deterministic_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "b_section").mkdir()
    (tmp_path / "a_section").mkdir()
    (tmp_path / "a_section" / "notes.md").write_text("a section notes")
    (tmp_path / "b_section" / "guide.txt").write_text("b section guide")
    (tmp_path / "b_section" / "diagram.png").write_bytes(b"not text")

    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path)
    documents = list(adapter.discover())

    assert [doc.relative_path for doc in documents] == [
        "a_section/notes.md",
        "b_section/guide.txt",
    ]
    assert documents[0].content == "a section notes"
    assert all(doc.source_id == "docs" for doc in documents)


def test_discover_excludes_symlinked_files_and_directories(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_file = real_dir / "runbook.md"
    real_file.write_text("real content")

    (tmp_path / "shortcut.md").symlink_to(real_file)
    (tmp_path / "linked").symlink_to(real_dir, target_is_directory=True)

    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path)
    documents = list(adapter.discover())

    assert [doc.relative_path for doc in documents] == ["real/runbook.md"]


def test_discover_raises_for_oversized_file(tmp_path: Path) -> None:
    (tmp_path / "big.md").write_text("x" * 50)
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path, max_file_size_bytes=10)

    with pytest.raises(SourceFileTooLargeError):
        list(adapter.discover())


def test_discover_raises_for_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / "broken.md").write_bytes(b"\xff\xfe not valid utf-8")
    adapter = LocalFilesystemSourceAdapter(source_id="docs", root=tmp_path)

    with pytest.raises(SourceReadError):
        list(adapter.discover())


@pytest.mark.parametrize("kind", ["missing", "not_a_directory"])
def test_constructor_rejects_missing_or_non_directory_root(tmp_path: Path, kind: str) -> None:
    if kind == "missing":
        root = tmp_path / "does-not-exist"
    else:
        root = tmp_path / "not-a-directory.md"
        root.write_text("x")

    with pytest.raises(SourceConfigurationError):
        LocalFilesystemSourceAdapter(source_id="docs", root=root)


def test_raw_document_and_source_do_not_leak_absolute_root_path(tmp_path: Path) -> None:
    (tmp_path / "deploy").mkdir()
    (tmp_path / "deploy" / "aegis.md").write_text("rollback steps")

    adapter = LocalFilesystemSourceAdapter(source_id="runbooks", root=tmp_path)
    documents = list(adapter.discover())

    assert len(documents) == 1
    assert documents[0].relative_path == "deploy/aegis.md"
    assert str(tmp_path) not in documents[0].relative_path

    source = adapter.source()
    assert source.source_id == "runbooks"
    assert source.source_type == "local_filesystem"


def test_document_identity_is_stable_across_different_root_locations(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    root_a = tmp_path_factory.mktemp("checkout_a")
    root_b = tmp_path_factory.mktemp("checkout_b")
    (root_a / "deploy").mkdir()
    (root_a / "deploy" / "aegis.md").write_text("first checkout")
    (root_b / "deploy").mkdir()
    (root_b / "deploy" / "aegis.md").write_text("second checkout, different content")

    doc_a = next(iter(LocalFilesystemSourceAdapter(source_id="runbooks", root=root_a).discover()))
    doc_b = next(iter(LocalFilesystemSourceAdapter(source_id="runbooks", root=root_b).discover()))

    assert derive_document_id(doc_a.source_id, doc_a.relative_path) == derive_document_id(
        doc_b.source_id, doc_b.relative_path
    )
