"""Errors for the chunking layer.

Chunking is a set of pure functions over text the caller already has in
hand, not a boundary talking to the outside world, so it needs nothing like
the source adapter's exception hierarchy. This exists for caller-side
configuration mistakes only, such as a non-positive ``max_chunk_chars``.

Document content itself never raises this: source documents are untrusted
and arbitrarily shaped, so malformed, unsupported, or structurally empty
Markdown is handled by falling back to treating the offending line as
ordinary content, not by failing the whole document.
"""

from __future__ import annotations


class ChunkingError(ValueError):
    """A nonsensical chunking configuration value, never a document-content problem."""
