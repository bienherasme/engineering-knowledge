"""Lexical retrieval boundary: the port hybrid retrieval will later consume.

``LexicalIndex`` is deliberately one method. Index maintenance (inserting or
removing FTS rows as documents change) is a persistence-adapter concern, not
part of this port: a caller of ``LexicalIndex`` only ever wants to search,
never to reach in and mutate the index directly.

BM25 is a strategy-specific relevance score, not a 0..1 confidence value and
not comparable across retrieval strategies. Hybrid retrieval will fuse
ranks (reciprocal rank fusion), not raw scores, so ``bm25_score`` is kept
exactly as SQLite's ``bm25()`` reports it rather than normalized into
something that only looks portable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from engineering_knowledge.domain import Chunk, SourceReference
from engineering_knowledge.retrieval.errors import InvalidQueryError

DEFAULT_MAX_RESULTS = 10
MAX_RESULTS = 50


@dataclass(frozen=True, slots=True)
class LexicalMatch:
    """One ranked lexical result, with structured provenance already attached.

    ``rank`` is 1-based and reflects this result's position in the ranked
    output, not a database row id. ``bm25_score`` is FTS5's native score:
    lower is a better match, and it has no meaning outside this retrieval
    strategy.
    """

    chunk: Chunk
    source_reference: SourceReference
    rank: int
    bm25_score: float


class LexicalIndex(Protocol):
    """Searches normalized chunk text using SQLite FTS5 BM25 ranking."""

    def search(
        self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS
    ) -> tuple[LexicalMatch, ...]: ...


def build_match_expression(query: str) -> str:
    """Convert ordinary user query text into a safe FTS5 MATCH expression.

    The query is treated as literal data, never as FTS query grammar: it is
    split on whitespace into terms, each term is quoted as an FTS string
    literal (with any embedded double quote doubled, FTS5's own escaping
    rule), and the quoted terms are joined with OR. Quoting means a term
    containing a hyphen, parenthesis, colon, or another FTS operator
    character is searched for as literal text instead of being interpreted
    as query syntax, so arbitrary user text cannot change the shape of the
    query or raise a syntax error.

    OR rather than the FTS default AND-between-terms is deliberate: lexical
    retrieval here is a candidate generator for a future hybrid strategy,
    so it should surface a chunk that matches some of a natural-language
    query's words, not only one that matches every word. BM25 still ranks
    chunks that match more terms higher.
    """
    stripped = query.strip()
    if not stripped:
        raise InvalidQueryError("query must not be blank")

    terms = stripped.split()
    escaped_terms = ['"' + term.replace('"', '""') + '"' for term in terms]
    return " OR ".join(escaped_terms)
