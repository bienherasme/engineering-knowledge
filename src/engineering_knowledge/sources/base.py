"""Source boundary: where document content enters the system from outside.

The domain layer knows nothing about the filesystem, network, or any other
external system. Everything in this package exists to translate one external
source into `RawDocument` instances, the only shape the rest of the pipeline
is allowed to see coming from a source. Normalization, chunking, and
identity derivation happen after this boundary, never inside it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from engineering_knowledge.domain import DocumentSource
from engineering_knowledge.domain.identity import normalize_relative_path


class SourceError(Exception):
    """Base class for expected, source-boundary failures.

    Anything that is not one of these, a bug in this codebase rather than an
    ordinary external problem, is left to propagate instead of being folded
    into this hierarchy.
    """


class SourceConfigurationError(SourceError):
    """The adapter itself is misconfigured, independent of any single document."""


class SourceReadError(SourceError):
    """Discovering or reading a specific document failed for an expected reason."""


class SourceFileTooLargeError(SourceReadError):
    """A candidate file exceeds the adapter's configured size bound."""


class RawDocument(BaseModel):
    """Content received from a source adapter, before normalization.

    Deliberately thinner than `Document`: no document_id, no content_hash,
    no ingestion timestamp, no document_type. Those are produced by the
    future normalization/ingestion step, not by reading a file.
    `relative_path` is required to already be the canonical output of
    `normalize_relative_path`, the same rule `Document.relative_path`
    enforces, so a path means the same thing everywhere it is derived from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    relative_path: str
    content: str

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        normalized = normalize_relative_path(value)
        if normalized != value:
            raise ValueError(
                f"relative_path must already be normalized: {value!r} != {normalized!r}"
            )
        return value


class SourceAdapter(Protocol):
    """A boundary that discovers and reads documents from one external source.

    `discover` is the only content-producing method on purpose. Splitting
    discovery from reading only pays for itself once an adapter needs to
    select among documents before paying the cost of fetching them, which no
    v0 source needs. A generator-based implementation can still fetch
    content lazily, one item at a time, inside `discover` itself, so this
    does not block a future remote source from being added without
    redesigning the port.
    """

    def source(self) -> DocumentSource:
        """The normalized identity of the source this adapter reads from."""
        ...

    def discover(self) -> Iterable[RawDocument]:
        """Yield every document currently available from this source.

        Implementations must fail the whole call with a `SourceError`
        rather than silently return a partial result if discovery cannot be
        completed.
        """
        ...
