"""Deterministic text normalization.

The only concern here is line-ending normalization. Everything else about
the source text, whitespace, blank lines, casing, code indentation, is left
exactly as written: those are meaningful content decisions that belong to
the author of the document, not to this pipeline. The normalized text this
function returns is what a future ingestion step hashes to produce
Document.content_hash, so normalizing more than this would silently change
what "the same document" means.
"""

from __future__ import annotations


def normalize_text(text: str) -> str:
    """Normalize CRLF and bare CR line endings to LF.

    Idempotent: normalizing already-normalized text returns it unchanged,
    since no CR characters remain after the first pass.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")
