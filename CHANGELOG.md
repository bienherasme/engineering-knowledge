# Changelog

## 0.1.0

Local-first engineering knowledge retrieval: deterministic ingestion,
provenance, hybrid search, and measurable retrieval evaluation.

**Ingestion and persistence**

- Sandboxed local Markdown/plain-text filesystem source adapter: root-contained,
  no symlink following, extension allowlist, bounded reads, UTF-8 only.
- Deterministic text normalization and section-aware Markdown chunking.
- Stable, content-independent document and chunk identity, with
  `content_hash`, `processing_fingerprint`, and `embedding_fingerprint` kept
  as separate, independent invalidation axes.
- SQLite normalized persistence with atomic, idempotent source
  synchronization; a full snapshot of one source's documents and chunks
  written in a single transaction, with deletions driven by discovery.

**Retrieval**

- FTS5 lexical retrieval with BM25 ranking.
- Optional `sqlite-vec` vector retrieval using a local Sentence
  Transformers embedding provider.
- Deterministic hybrid retrieval via Reciprocal Rank Fusion.
- One active vector embedding configuration per database, changed only
  through an explicit vector rebuild; source-scoped ingestion can maintain
  a compatible active index but never repoints it.
- A public `KnowledgeService` with explicit `SUCCESS` / `EMPTY` / `PARTIAL`
  retrieval outcomes and full source provenance on every result.

**Evaluation**

- A synthetic five-document engineering corpus with genuine cross-document
  overlap and ambiguity.
- A versioned golden retrieval dataset using logical chunk references,
  resolved to chunk IDs at evaluation time.
- A Recall@K / MRR evaluation harness that runs lexical, vector, and
  hybrid retrieval through the same public `KnowledgeService` used by the
  CLI and MCP adapter.
- A versioned baseline evaluation artifact measured with real embeddings.

**Adapters**

- A CLI (`engineering-knowledge`) for ingestion, search, vector index
  maintenance, and evaluation, with optional JSON output.
- A read-only, stdio-only MCP adapter exposing exactly three knowledge
  tools (`search_knowledge`, `get_document`, `get_chunk`) over an
  already-prepared database, with no ingestion, rebuild, or mutation
  capability.
- A read-only SQLite serving mode used by the MCP adapter: requires an
  existing, current-schema database and performs no migration.

**Configuration**

- A small typed TOML configuration, with filesystem paths resolved
  relative to the configuration file rather than the process working
  directory.
