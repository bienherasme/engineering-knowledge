# Engineering Knowledge

## Purpose

Engineering Knowledge is a retrieval system for technical knowledge used by engineering teams.

The system is intended to make architectural decisions, runbooks, postmortems, API documentation, operational procedures, and other engineering references searchable and usable by AI-assisted engineering workflows.

## v0.1.0 Scope

In scope:

- local Markdown and plain-text sources
- deterministic, section-aware chunking
- SQLite-backed persistence
- FTS5 lexical retrieval
- vector retrieval using a local real embedding provider
- hybrid retrieval via reciprocal rank fusion
- provenance and source citations
- retrieval evaluation harness (Recall@K, MRR)
- a library API and CLI
- a thin MCP adapter over the same library API
- a synthetic corpus and golden evaluation dataset

Out of scope for v0.1.0:

- PDF and other non-text source formats
- remote source integrations (Drive, Confluence, Notion, SharePoint, Slack, GitHub API)
- answer generation
- LLM-based reranking
- cross-encoder reranking
- a chatbot UI
- a cloud-hosted vector database
- autonomous or multi-agent orchestration
- the Incident Commander integration itself

Engineering Knowledge must remain independently useful without Incident Commander
or any other consumer.

## Architectural Decisions

These are treated as settled constraints, not open questions, for v0.1.0 design work.

1. **Fake embeddings do not validate retrieval quality.** A deterministic
   hash-to-vector provider is for unit tests and persistence/index plumbing only.
   It has no semantic content, so it cannot be used to evaluate vector or hybrid
   retrieval. The evaluation harness that compares lexical, vector, and hybrid
   strategies must use a real local embedding provider.

2. **Cross-project integration goes through an external boundary, not a direct
   import.** Engineering Knowledge exposes a reusable Python core, but the
   intended integration with Incident Commander is Incident Commander's own
   `KnowledgeContext` port talking to Engineering Knowledge as an external
   system, most likely over MCP. Direct library usage stays available for the
   CLI, the evaluation harness, and same-process consumers that explicitly want
   that coupling, but it is not the recommended integration path between the
   two projects.

3. **Logical identity is separate from content version.** Document identity is
   derived from a configured stable `source_id` plus a normalized relative
   path, never an absolute local checkout path. Chunk identity is a logical,
   deterministic identity derived from `document_id` plus section identity and
   a stable ordinal. `content_hash` is a separate field identifying which
   version of that chunk's content was ingested, and does not participate in
   chunk identity. `ingested_at` does not participate in identity either.

4. **Retrieval outcomes and errors are not the same axis.** `RetrievalResult`
   carries a `status` of `SUCCESS`, `EMPTY`, or `PARTIAL` for outcomes of a
   valid retrieval attempt (`PARTIAL` requires a `truncation_reason`). A
   malformed query or filter is an `InvalidQueryError`, and an unavailable or
   corrupt index is an `IndexUnavailableError`. Caller mistakes and
   infrastructure failures are never represented as ordinary result statuses.

5. **sqlite-vec is the target vector persistence, not yet a frozen dependency.**
   It must first pass a small spike confirming installation, the Python
   versions this repo targets, persistent storage, insert/update/delete,
   nearest-neighbor query, and metadata filtering, in a clean local setup. If
   it fails materially, only the vector index adapter gets replaced; the
   surrounding architecture does not change.

## Knowledge Sources

Initial source types may include:

- architecture decision records
- runbooks
- postmortems
- API documentation
- engineering standards
- system design documents

The initial implementation will use synthetic or public sample documents rather than private employer data.

## Architecture Direction

The system will separate:

- ingestion
- document normalization
- indexing
- retrieval
- reranking
- answer generation
- evaluation

Retrieval quality should be measurable independently from final LLM output.

## Design Principles

- source-grounded answers
- explicit citations
- metadata-aware retrieval
- retrieval evaluation
- no silent invention of missing knowledge
- clear distinction between retrieved facts and generated explanation
- provider-independent domain models where practical

## Out of Scope Initially

- company-specific proprietary knowledge
- autonomous document modification
- unrestricted enterprise search
- access-control architecture for a multi-tenant SaaS product
- large-scale distributed indexing
- fine-tuning

## Relationship to Other Personal Projects

The system may later provide technical knowledge such as runbooks and postmortems to independent incident-response or architecture-analysis workflows.

That integration is expected to happen through an external boundary, most likely MCP, with the consuming project owning its own port into this system, rather than through a direct Python import. See Architectural Decisions above.

## Project Origin

This project concept and its initial scope were defined before the start of my next employment engagement.