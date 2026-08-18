"""Errors for the retrieval boundary, lexical and vector.

Independent failure categories, deliberately not merged into one hierarchy:
a bad query is a caller mistake, an unavailable index is an infrastructure
problem, and an incompatible index is neither, it is a valid query against a
persisted index that answers a different question than the one asked.
Collapsing these would make a caller unable to tell "you asked for something
invalid" from "the system could not answer even a valid question" from "this
answer would be meaningless", which matters for how each should be handled
upstream.

``InvalidQueryError`` is shared between lexical and vector retrieval: a
blank query or a bad ``max_results`` means the same thing regardless of
strategy, so there is exactly one caller-input error type for it, not one
per retriever.
"""

from __future__ import annotations


class InvalidQueryError(ValueError):
    """A caller-supplied query or bound is invalid.

    Covers a blank query and a non-positive or excessive max_results.
    """


class IndexUnavailableError(Exception):
    """A required lexical-search index structure is unavailable at query time."""


class VectorIndexUnavailableError(Exception):
    """Vector search infrastructure is unavailable: not enabled, not built, or failed to load."""


class VectorIndexIncompatibleError(Exception):
    """A vector or embedding fingerprint does not match the active vector index.

    Raised both at query time (a query vector from a different provider/
    model than the active index) and at source-sync time (ingestion
    supplies vectors under a different fingerprint than the active index,
    or supplies none at all while the active index requires them for a
    document that needs writing). In every case the fix is the same:
    rebuild_vector_index, never an automatic change during ordinary sync.
    """


class HybridFusionError(ValueError):
    """A component retriever violated the contract fusion relies on.

    Covers a component returning the same chunk_id more than once, and two
    components disagreeing about the content of a chunk_id they both
    returned. Neither is a caller mistake or an infrastructure outage; both
    mean a retriever implementation broke an invariant fusion assumes
    without checking on every call.
    """
