# Architecture

This document goes deeper than the README. It focuses on invariants and
tradeoffs, not a class-by-class reference; read the source for exact
signatures.

## System boundary

Engineering Knowledge is a retrieval and knowledge-serving library with two
thin adapters (CLI, MCP) over one public service, `KnowledgeService`. It
ingests local text sources, normalizes and chunks them deterministically,
persists normalized state in SQLite, derives lexical and (optionally)
vector indexes from that state, and answers queries with ranked,
provenance-carrying results. It does not generate answers, rerank with a
model, or talk to anything remote. There is no LLM in the retrieval path.

## Layering

```
domain (leaf: models, identity)
  ^
  |
chunking, sources, embeddings, retrieval (mid: no dependency on persistence)
  ^
  |
persistence (implements the retrieval/repository ports)
  ^
  |
ingestion, evaluation, cli, mcp (top: orchestration and adapters)
```

`retrieval` never imports `persistence`; it defines the small
`KnowledgeRepository`/`LexicalIndex`/`VectorIndex` ports it needs, and
`SqliteRepository` satisfies them structurally. This keeps the retrieval
package's public contract independent of any specific storage engine, and
it is verified empirically (a fresh-process import check), not just by
convention.

## Source boundary

`SourceAdapter` is the only place external content enters the system.
Everything past it (normalization, chunking, identity) sees only
`RawDocument`: source id, a normalized relative path, and text content. The
one implementation in v0.1.0, `LocalFilesystemSourceAdapter`, treats
everything under its configured root as untrusted: paths must resolve
inside the root, symlinked files and directories are excluded outright
rather than validated, file size and UTF-8 encoding are enforced before
content is handed upward, and directory traversal is deterministic
(sorted) so discovery order never depends on filesystem iteration order.

## Domain identity

Two identities, deliberately independent of content:

- **`document_id`**: derived from `(source_id, normalized relative_path)`.
  The same logical document keeps the same id across content edits.
- **`chunk_id`**: derived from `(document_id, section identity, section
  occurrence, ordinal within that section occurrence)`.

Neither includes a content hash or a timestamp. `section_occurrence` exists
because a document can legitimately contain two sections with the same
human-visible heading path (two "## Examples" sections, say); without it
their chunks would collide on identity. It counts occurrences of the same
canonical section path in document order. A chunk's identity is stable
under edits to its own text and under changes to unrelated sections
elsewhere in the document, because no global ordinal participates in it.

Three separate fingerprints answer three separate questions, and the
system never conflates them:

- **`content_hash`**: which version of this chunk's (or document's)
  content was ingested. Participates in change detection, never in identity.
- **`processing_fingerprint`**: which normalization/chunking *behavior*
  produced a document's chunks (normalization version, chunking version,
  document format, `max_chunk_chars`, chunk overlap). Changing
  `max_chunk_chars` changes this fingerprint and causes reprocessing; it
  never changes `document_id` or a surviving chunk's `chunk_id`.
- **`embedding_fingerprint`**: which embedding *representation* produced a
  chunk's vector (provider type, model id, model revision, dimension, the
  embedding-input text format's own version). Changing the embedding model
  changes this fingerprint and requires a vector rebuild; it never touches
  `processing_fingerprint` or causes a document to be reported as updated.

Keeping these three axes independent is what lets `max_chunk_chars` and the
embedding model each change on their own schedule without either
invalidating the other's work, and without either ever looking like a
content edit.

## Normalization and chunking

Text is normalized deterministically (line endings, trailing whitespace)
before chunking, so the same source content always produces the same
normalized text regardless of how it was saved. Chunking is section-aware:
ATX Markdown headings define a heading stack, fenced code blocks are
tracked so a `#` inside a code fence is never mistaken for a heading, and
chunks never overlap in v0.1.0. A heading line that does not resolve to a
real title (an empty or malformed ATX-looking line) falls back to ordinary
content rather than failing ingestion.

## Atomic source synchronization

One call to `sync_source` represents one attempt to bring persisted state
for a single configured source fully in line with what that source
currently contains. Discovery and processing (and, when an embedding
provider is configured, embedding) happen entirely in memory before any
write: a failure partway through never partially mutates persisted state.
A successful discovery is authoritative for deletions: any previously
persisted document for that source not present in the current discovery is
removed, chunks included. The normalized tables and the derived FTS5 index
update together inside the same SQLite transaction; when a vector index is
active, its rows update in the same transaction too. This is one SQLite
database's transactional guarantee, not a distributed or cross-database
transaction.

Reprocessing an unchanged document (same `content_hash`, same
`processing_fingerprint`) is a genuine no-op: the row, including its
ingestion timestamp, is left untouched, and its vectors, if any, are left
untouched. Reindexing an unchanged document under a new embedding
configuration is `rebuild_vector_index`'s job, not ordinary sync's.

## Authoritative state vs. derived indexes

`documents` and `chunks` in SQLite are the single authoritative normalized
state. The FTS5 virtual table and the `sqlite-vec` virtual table are both
fully derived from that state and can be rebuilt from it at any time;
neither is a second source of truth, and neither is ever read without the
normalized tables agreeing with it in the same transaction that wrote it.

## Lexical retrieval

FTS5 with `unicode61`, `_` and `-` added as token characters and no
stemming: this system's queries look like engineering identifiers
(`MAX_RETRY_COUNT`, `payments-service`), and splitting on every underscore
or hyphen, or stemming an identifier as if it were a natural-language word,
would make exact-identifier search worse, not more flexible. Query text is
converted to a safe FTS MATCH expression before it ever reaches SQLite;
raw FTS syntax from a caller is never interpreted as a query language.

## Vector retrieval and index configuration

A database has **one active embedding configuration** at a time:
persisted `embedding_fingerprint` and dimension, plus the `sqlite-vec`
table sized for that dimension. Only `rebuild_vector_index(provider)` may
create or change that active configuration; it reads every persisted
chunk directly (no source re-ingestion required), computes fresh vectors,
and atomically replaces the vector table and its metadata.

Source-scoped ingestion (`sync_source`) may *maintain* an already-active,
compatible vector index (if it is supplied vectors under the same active
fingerprint, it writes them alongside the normalized rows in the same
transaction), but it can never change the active fingerprint or dimension,
and it can never activate vector indexing on its own. Ingesting with a
mismatched or missing embedding configuration while a vector index is
active fails explicitly (`VectorIndexIncompatibleError`) rather than
silently degrading, silently leaving vectors stale, or repointing the
global configuration out from under other sources sharing the same
database. This is what makes multi-source ingestion safe under embedding
model changes: no ordinary sync of one source can corrupt or shift the
vector space another source's chunks already live in.

Query time enforces the same rule the other direction: a query vector's
fingerprint must match the active index's fingerprint, or the query fails
explicitly rather than comparing vectors from different embedding spaces
and returning meaningless distances.

## Hybrid retrieval

Deterministic Reciprocal Rank Fusion over the lexical and vector rankings,
`score += 1 / (RRF_K + rank)` per component with `RRF_K = 60`, summed
across whichever components returned a given chunk. Fusion operates on
rank, never on raw BM25 or distance values, which are on incomparable
scales (BM25 and vector distance are lower-is-better on their own scales;
RRF score is higher-is-better; none of the three are ever mixed or renamed
into a generic "relevance"). A component returning the same chunk id twice,
or two components disagreeing about a chunk id's content, is a contract
violation (`HybridFusionError`), not something fusion silently tolerates.

## Public retrieval result semantics

`KnowledgeService.search` returns a `RetrievalResult` with one of three
statuses, all of them *valid outcomes* of a valid query:

- `SUCCESS`: one or more results, not known to be truncated.
- `EMPTY`: a valid query matched nothing.
- `PARTIAL`: one or more results, and the public `max_results` bound is
  known to have truncated further ranked results (`truncation_reason` is
  always set). Truncation is detected by requesting one extra result
  internally and checking whether it exists, never guessed from `len(results)
  == max_results`.

`PARTIAL` means exactly "more results exist past the requested bound." It
never means degraded infrastructure. Requesting `VECTOR` or `HYBRID`
without a configured vector retriever raises `VectorIndexUnavailableError`
outright; there is no silent fallback to lexical.

## Failure semantics

Three independent categories, never collapsed into one:

- **Caller error**: `InvalidQueryError` (blank query, invalid
  `max_results`). The caller asked for something invalid.
- **Infrastructure/capability error**: persistence, index, or vector
  errors (`PersistenceError`, `IndexUnavailableError`,
  `VectorIndexUnavailableError`, `VectorIndexIncompatibleError`,
  `EmbeddingError`, `ConfigurationError`). The system could not answer even
  a valid request.
- **Valid retrieval outcome**: `SUCCESS`/`EMPTY`/`PARTIAL`. The system
  answered; this is what it found.

Expected source, configuration, and persistence failures are typed
exceptions a caller can catch specifically. Unexpected programming errors
are never caught and relabeled as `EMPTY` or as an ordinary successful
result; they propagate.

## Evaluation architecture

The evaluation harness (`engineering_knowledge.evaluation`) is a thin
measurement layer, not a second retrieval implementation. Golden relevance
judgments are stored as logical references (source id, relative path,
section path, section occurrence, ordinal), resolved to chunk ids at
evaluation time through the same `derive_document_id`/`derive_chunk_id`
helpers ingestion itself uses, never hardcoded opaque ids, so a stale
reference fails loudly (`EvaluationDatasetError`) instead of silently
scoring against the wrong or no content. Every query is evaluated by
calling the public `KnowledgeService.search`, exactly the boundary the CLI
and MCP adapters use, for each of the three strategies; metrics
(`recall_at_k`, `reciprocal_rank`) are pure functions over the returned
ordered chunk ids, independent of SQLite, embeddings, or the service
itself. An infrastructure exception during evaluation propagates and fails
the run; it is never converted into a zero score, which would misrepresent
"the system could not answer" as "the system answered and found nothing."

## CLI and MCP adapter boundaries

Both adapters compose already-built services through
`engineering_knowledge.composition`'s small explicit factory functions;
neither reimplements ingestion, retrieval, or evaluation logic. Composition
is capability-specific and lazy: a lexical-only command never constructs an
embedding provider, and building a knowledge-serving MCP session never
opens a writable repository.

The CLI is a maintenance-capable adapter (`ingest`, `search`,
`rebuild-vectors`, `evaluate`, `serve-mcp`); it is where writes happen. The
MCP adapter is deliberately narrower: exactly three read-only tools
(`search_knowledge`, `get_document`, `get_chunk`), no ingest, rebuild,
evaluate, filesystem, or SQL surface, stdio transport only. Only
`engineering_knowledge/mcp/server.py` imports the `mcp` SDK; nothing in
`domain`, `ingestion`, `persistence`, `retrieval`, or `evaluation` does,
and the base package imports cleanly without the optional `mcp` dependency
installed.

## Read-only MCP serving

`SqliteRepository(read_only=True)` opens the database file through
SQLite's URI `mode=ro` semantics: a missing file is refused rather than
created, and a schema version other than the current one is a hard stop
rather than a migration attempt, since a read-only connection cannot write
one. `sync_source` and `rebuild_vector_index` fail immediately at this
boundary (`PersistenceError`) rather than relying on SQLite eventually
rejecting a write deep inside a transaction. MCP composition always uses
this mode: knowledge serving depends only on already-persisted state, never
on source availability, and it never ingests, migrates, or rebuilds
anything, at startup or at query time.

One implementation detail worth naming: MCP tool functions are declared
`async def`, not because they await anything, but because the SDK offloads
plain synchronous tool functions to a worker thread pool to avoid blocking
its event loop. A `sqlite3` connection is bound to the thread that created
it, so running a query from a different worker thread breaks outright;
`async def` keeps every tool call on the thread that opened the connection.

## Optional dependency boundaries

Three independent optional extras, each imported lazily and only where it
is actually needed:

- **`vector`** (`sqlite-vec`): imported only inside
  `SqliteRepository._load_vector_extension`, called only when a repository
  is constructed with `vector_index_enabled=True`. Extension loading is
  enabled immediately before that one load and disabled immediately after.
- **`local-embeddings`** (`sentence-transformers`): imported only inside
  `SentenceTransformersEmbeddingProvider.__init__`, constructed only when a
  caller actually needs a real embedding provider.
- **`mcp`**: imported only inside the MCP adapter module, loaded lazily by
  the CLI's `serve-mcp` command.

The base install (`pip install -e .`) never imports any of the three, and
lexical ingestion and search never require any of them.

## Extension points

The layering is designed so a future addition slots in without touching
what already works:

- A new `SourceAdapter` (a remote source, a different file format) only
  needs to produce `RawDocument`s; normalization, chunking, identity, and
  persistence are unaffected.
- A new `EmbeddingProvider` only needs to implement the small
  `embed_documents`/`embed_query` port; fingerprinting and vector index
  configuration are provider-agnostic.
- A new retrieval strategy would extend `RetrievalStrategy` and
  `KnowledgeService.search`'s dispatch, without changing `RetrievalResult`'s
  shape or the CLI/MCP adapters that already consume it.
- A new adapter (an HTTP API, a different protocol) would compose from
  `engineering_knowledge.composition` exactly as the CLI and MCP adapters
  do, never reimplementing retrieval.

None of these are implemented in v0.1.0; they are named here because the
layering was deliberately chosen to make them additive later.
