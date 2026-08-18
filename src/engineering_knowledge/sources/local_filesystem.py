"""Local filesystem source adapter.

Reads a bounded, extension-filtered set of text files from a configured
root directory. The root itself is trusted configuration; everything found
beneath it is untrusted input and is treated accordingly: every candidate
must resolve inside the root, symlinks are excluded rather than validated,
and both file size and text encoding are enforced before content is handed
to the caller.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from engineering_knowledge.domain import DocumentSource
from engineering_knowledge.domain.identity import normalize_relative_path
from engineering_knowledge.sources.base import (
    RawDocument,
    SourceConfigurationError,
    SourceFileTooLargeError,
    SourceReadError,
)

DEFAULT_ALLOWED_EXTENSIONS = frozenset({".md", ".txt"})
DEFAULT_MAX_FILE_SIZE_BYTES = 1_000_000

_SOURCE_TYPE = "local_filesystem"


class LocalFilesystemSourceAdapter:
    """Discovers and reads text documents from a local directory tree.

    Symlinked files and directories are excluded from discovery entirely
    rather than followed or individually validated: a local corpus has no
    legitimate need for them, and refusing them outright removes a whole
    class of path-escape risk instead of trying to safely validate it away.

    Hidden files and directories (dotfiles) are not treated specially and
    are included when they match an allowed extension. "Hidden" is a
    filename convention, not a security property; containment, the symlink
    policy, the extension allowlist, and the size bound are what actually
    keep discovery safe.
    """

    def __init__(
        self,
        *,
        source_id: str,
        root: Path,
        allowed_extensions: frozenset[str] = DEFAULT_ALLOWED_EXTENSIONS,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        display_name: str | None = None,
    ) -> None:
        if not source_id.strip():
            raise SourceConfigurationError("source_id must not be empty")
        if max_file_size_bytes <= 0:
            raise SourceConfigurationError("max_file_size_bytes must be positive")
        if not root.exists():
            raise SourceConfigurationError(f"source root does not exist: {root}")
        if not root.is_dir():
            raise SourceConfigurationError(f"source root is not a directory: {root}")

        self._source_id = source_id
        # Resolved once, kept only for internal containment checks. This is
        # a filesystem security boundary, unlike domain identity path
        # normalization, so resolving symlinks and `..` here is correct.
        # The resolved path never leaves this adapter or reaches a domain
        # model.
        self._root = root.resolve()
        self._allowed_extensions = frozenset(ext.lower() for ext in allowed_extensions)
        self._max_file_size_bytes = max_file_size_bytes
        self._display_name = display_name

    def source(self) -> DocumentSource:
        return DocumentSource(
            source_id=self._source_id,
            source_type=_SOURCE_TYPE,
            display_name=self._display_name,
        )

    def discover(self) -> Iterator[RawDocument]:
        for candidate, relative_path in self._discover_candidates():
            yield self._read(candidate, relative_path)

    def _discover_candidates(self) -> list[tuple[Path, str]]:
        def _on_walk_error(error: OSError) -> None:
            # os.walk silently drops unreadable subdirectories by default.
            # That is exactly the silent-partial-discovery outcome we do
            # not want, so any traversal error fails the whole call.
            raise SourceReadError(f"failed to list directory: {error.filename}") from error

        candidates: list[tuple[Path, str]] = []
        for dirpath, dirnames, filenames in os.walk(
            self._root, onerror=_on_walk_error, followlinks=False
        ):
            current_dir = Path(dirpath)
            # followlinks=False already stops os.walk from recursing into a
            # symlinked directory, but it still lists that directory in
            # dirnames for the current level. Dropping it here makes the
            # policy explicit rather than relying on that implicit contract.
            dirnames[:] = sorted(
                name for name in dirnames if not (current_dir / name).is_symlink()
            )
            for filename in sorted(filenames):
                candidate = current_dir / filename
                if candidate.is_symlink():
                    continue
                if not candidate.is_file():
                    continue
                if candidate.suffix.lower() not in self._allowed_extensions:
                    continue
                candidates.append((candidate, self._relative_path_for(candidate)))

        candidates.sort(key=lambda item: item[1])
        return candidates

    def _relative_path_for(self, candidate: Path) -> str:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self._root):
            raise SourceReadError(f"file escapes configured source root: {candidate}")
        return normalize_relative_path(resolved.relative_to(self._root).as_posix())

    def _read(self, candidate: Path, relative_path: str) -> RawDocument:
        try:
            size_bytes = candidate.stat().st_size
        except OSError as error:
            raise SourceReadError(f"failed to stat file: {relative_path}") from error

        if size_bytes > self._max_file_size_bytes:
            raise SourceFileTooLargeError(
                f"file exceeds max size of {self._max_file_size_bytes} bytes: {relative_path}"
            )

        try:
            with candidate.open("rb") as handle:
                # Bounded read, not stat-then-trust: a file can grow between
                # the stat above and this read, so the real bound is that
                # this read never pulls in more than one byte past the
                # limit, regardless of what stat reported.
                raw_bytes = handle.read(self._max_file_size_bytes + 1)
        except OSError as error:
            raise SourceReadError(f"failed to read file: {relative_path}") from error

        if len(raw_bytes) > self._max_file_size_bytes:
            raise SourceFileTooLargeError(
                f"file exceeds max size of {self._max_file_size_bytes} bytes: {relative_path}"
            )

        try:
            # utf-8-sig transparently strips a leading UTF-8 BOM if present
            # and behaves exactly like utf-8 otherwise.
            content = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise SourceReadError(f"file is not valid UTF-8: {relative_path}") from error

        return RawDocument(
            source_id=self._source_id,
            relative_path=relative_path,
            content=content,
        )
