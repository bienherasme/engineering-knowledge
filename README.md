# Engineering Knowledge

Local-first engineering knowledge retrieval with deterministic ingestion,
provenance, hybrid search, and measurable retrieval evaluation.

## Why this exists

Most "chat with your docs" projects hide retrieval quality behind an LLM
answer, so a wrong or missing citation is invisible until it causes real
damage. Engineering Knowledge treats retrieval as the product: every
document and chunk has a stable, deterministic identity, every result
carries its source provenance, and retrieval quality is measured with
Recall@K and MRR against a versioned golden dataset, not judged by how
convincing a generated paragraph sounds. There is no LLM anywhere in the
retrieval path.

## Capabilities

- Local Markdown/plain-text ingestion through a sandboxed filesystem source
- Deterministic text normalization and section-aware chunking
- Stable, content-independent document/chunk identity with full provenance
- SQLite-backed normalized persistence with atomic, idempotent source sync
- FTS5 lexical retrieval
- Optional `sqlite-vec` vector retrieval with local Sentence Transformers embeddings
- Deterministic hybrid retrieval via Reciprocal Rank Fusion (RRF)
- A public `KnowledgeService` with explicit `SUCCESS` / `EMPTY` / `PARTIAL` outcomes
- A synthetic engineering corpus, a golden retrieval dataset, and a Recall@5/MRR harness
- A CLI for ingestion, search, vector maintenance, and evaluation
- A read-only MCP stdio adapter for serving an already-prepared database to MCP clients

Not included in v0.1.0: answer generation, a chatbot UI, reranking, document
filters, PDF or other non-text formats, remote source integrations, remote
MCP transport, an HTTP API, or a cloud vector database. See
[Limitations](#limitations--non-goals).

## Architecture

```
Local sources
    |
    v
SourceAdapter (sandboxed filesystem boundary)
    |
    v
Normalization + deterministic section-aware chunking
    |
    v
SQLite: documents + chunks (authoritative normalized state)
    |                          |
    v                          v
FTS5 lexical index       sqlite-vec vector index
   (derived)                 (derived, optional)
    \                          /
     \                        /
        Hybrid RRF fusion
              |
              v
        KnowledgeService (public retrieval boundary)
          /                \
         v                  v
        CLI            MCP stdio (read-only)
```

Documents and chunks in SQLite are the single authoritative state. The FTS5
and vector indexes are both derived from that state and can be rebuilt from
it; neither is a second source of truth. `KnowledgeService` is the one
public entry point both adapters use, so the CLI and MCP server can never
diverge in retrieval behavior. See [docs/architecture.md](docs/architecture.md)
for the full design, including identity, fingerprinting, and failure
semantics.

## Retrieval strategies

- **Lexical**: SQLite FTS5, BM25-ranked.
- **Vector**: nearest-neighbor search over `sqlite-vec`, using a local
  Sentence Transformers embedding provider. Optional; requires the `vector`
  and `local-embeddings` extras and an explicit vector index build.
- **Hybrid**: deterministic Reciprocal Rank Fusion over the lexical and
  vector rankings (`RRF_K = 60`), never a blend of raw scores.

## Evaluation results

Measured on the repository's synthetic `aegis-demo` corpus (5 documents, 41
chunks) against a 12-query golden dataset (4 lexical / 4 semantic / 4 mixed),
using real `sentence-transformers/all-MiniLM-L6-v2` embeddings. See
[eval/baseline.json](eval/baseline.json) for the full, versioned artifact.

| Strategy | Recall@5 | MRR    |
|----------|----------|--------|
| Lexical  | 0.7083   | 0.6111 |
| Vector   | 0.7083   | 0.4333 |
| Hybrid   | 0.7917   | 0.5278 |

By query category:

| Category  | Strategy | Recall@5 | MRR    |
|-----------|----------|----------|--------|
| Lexical   | lexical  | 0.875    | 0.708  |
| Lexical   | vector   | 0.750    | 0.583  |
| Lexical   | hybrid   | 0.875    | 0.583  |
| Semantic  | lexical  | 0.750    | 0.563  |
| Semantic  | vector   | 1.000    | 0.542  |
| Semantic  | hybrid   | 1.000    | 0.667  |
| Mixed     | lexical  | 0.500    | 0.563  |
| Mixed     | vector   | 0.375    | 0.175  |
| Mixed     | hybrid   | 0.500    | 0.333  |

Reading these honestly: hybrid improved overall Recall@5 over either single
strategy on this corpus, but lexical retained the highest overall MRR
because it tends to rank its one strong hit first when the query shares
exact terminology with the document. Vector and hybrid both reached full
Recall@5 on the semantic category, where queries deliberately avoid the
corpus's exact wording. Mixed-category queries were the weakest for every
strategy. This is a small, synthetic, 5-document corpus built to exercise
retrieval behavior, not a general-purpose or production accuracy benchmark,
and these numbers should not be read as state-of-the-art or as evidence
about any other corpus.

### Evaluation integrity

- The corpus and golden dataset are versioned in the repository
  (`corpus/`, `eval/golden_dataset.json`).
- Golden relevance judgments are logical chunk references (source, path,
  section, occurrence, ordinal), resolved to chunk IDs at evaluation time
  through the same identity helpers ingestion uses, not hardcoded opaque IDs.
- The baseline was measured with a real embedding provider
  (`all-MiniLM-L6-v2`). The deterministic hash-based `FakeEmbeddingProvider`
  exists only for unit tests and index plumbing and is never used to
  produce or support a quality claim.
- Evaluation always goes through the public `KnowledgeService.search`, the
  same path the CLI and MCP adapter use, never an internal retriever directly.
- Metrics are exactly Recall@5 and MRR, macro-averaged over queries.
- No tuning, corpus editing, or query rewriting was performed after the
  first valid baseline was measured.
- With `max_chunk_chars = 1000`, the current demo corpus produces 41 chunks,
  and every section referenced by the golden dataset currently fits inside
  one chunk. That is a property of this specific small corpus, not a general
  chunking benchmark.

## Quick start

Base install, lexical retrieval only, no optional dependencies:

```bash
pip install -e .

engineering-knowledge ingest --config examples/engineering-knowledge.toml
engineering-knowledge search "PaymentGatewayTimeoutError" --config examples/engineering-knowledge.toml
```

The example config's paths are resolved relative to the config file itself,
so pointing `--config` directly at `examples/engineering-knowledge.toml`
from the repository root works without copying it; the database is written
to `data/engineering-knowledge.db` (git-ignored).

## Vector and hybrid retrieval

Vector and hybrid search require the optional extras and a real embedding
provider:

```bash
pip install -e ".[vector,local-embeddings]"
```

1. Enable embeddings in your config:

   ```toml
   [embeddings]
   enabled = true
   provider = "sentence_transformers"
   model_id = "sentence-transformers/all-MiniLM-L6-v2"
   ```

2. Ingest normalized state if you have not already:

   ```bash
   engineering-knowledge ingest --config engineering-knowledge.toml
   ```

3. Build the vector index:

   ```bash
   engineering-knowledge rebuild-vectors --config engineering-knowledge.toml
   ```

4. Query with vector or hybrid retrieval:

   ```bash
   engineering-knowledge search "rollback failed payment deploy" \
       --config engineering-knowledge.toml --strategy hybrid
   ```

Changing the embedding model or provider requires an explicit
`rebuild-vectors` run; it does not require re-ingesting the source. One
database has exactly one active vector configuration at a time, and only
`rebuild-vectors` may change it (see
[docs/architecture.md](docs/architecture.md#vector-index-configuration)).

## CLI

```
engineering-knowledge ingest          --config <path> [--json]
engineering-knowledge search <query>  --config <path> [--strategy lexical|vector|hybrid] [--max-results N] [--json]
engineering-knowledge rebuild-vectors --config <path>
engineering-knowledge evaluate        --config <path> --dataset <golden_dataset.json> [--k N] [--json]
engineering-knowledge serve-mcp       --config <path>
```

`--config` defaults to `./engineering-knowledge.toml`; a missing file is a
clear configuration error, not a silent fallback. `--json` on `ingest`,
`search`, and `evaluate` prints the underlying structured result as JSON to
stdout, suitable for scripting.

Run the same evaluation the published baseline used:

```bash
engineering-knowledge evaluate \
    --config engineering-knowledge.toml \
    --dataset eval/golden_dataset.json \
    --k 5
```

This is a read operation: it never edits the corpus, the golden dataset, or
the versioned `eval/baseline.json`. That file is a release artifact, not
something an ordinary evaluation run regenerates.

## MCP

```bash
pip install -e ".[mcp]"
engineering-knowledge serve-mcp --config engineering-knowledge.toml
```

Starts a **read-only**, **stdio-only** MCP server over an already-prepared
database. The tool surface is exactly:

- `search_knowledge`: delegates to `KnowledgeService.search`, returns a
  structured `RetrievalResult`.
- `get_document`: looks up one document by ID; returns `found: false`
  rather than an error when it does not exist.
- `get_chunk`: same, for one chunk.

There is no ingest, rebuild, evaluate, filesystem, or SQL tool, and no
generated answer. The database must already exist and be on the current
schema version; the server never migrates, ingests, or rebuilds anything at
startup or at query time. If `embeddings.enabled = false`, lexical search
works and vector/hybrid requests fail explicitly rather than falling back
silently. If embeddings are enabled, the configured provider must match the
persisted active vector fingerprint or vector/hybrid queries fail
explicitly with the same typed error the CLI uses.

## Configuration

A small typed TOML file, parsed with the standard library's `tomllib`:

```toml
[source]
source_id = "aegis-demo"
root = "./corpus"
max_file_size_bytes = 1000000

[persistence]
db_path = "./data/engineering-knowledge.db"

[processing]
max_chunk_chars = 1000

[retrieval]
default_strategy = "lexical"

[embeddings]
enabled = false
provider = "sentence_transformers"
model_id = "sentence-transformers/all-MiniLM-L6-v2"
```

Relative paths (`source.root`, `persistence.db_path`) are resolved relative
to the config file's own directory, never the process's working directory.
See [examples/engineering-knowledge.toml](examples/engineering-knowledge.toml).

## Optional dependencies

```bash
pip install -e .                                        # base: lexical only
pip install -e ".[dev]"                                  # + test/lint/type tooling
pip install -e ".[vector]"                                # + sqlite-vec
pip install -e ".[local-embeddings]"                       # + sentence-transformers
pip install -e ".[mcp]"                                    # + MCP stdio server
pip install -e ".[dev,vector,local-embeddings,mcp]"        # everything, local dev
```

Vector search requires a Python/SQLite build that supports loadable SQLite
extensions; some platform Python builds disable extension loading at
compile time. This is an environment note, not a project requirement:
`requires-python` remains `>=3.11`, and lexical/base operation never needs
`sqlite-vec` or extension loading at all.

## Trust and security boundaries

- Source content is treated as untrusted input, even though it currently
  comes from the local filesystem. The filesystem source adapter is
  root-contained, follows no symlinked files or directories, allow-lists
  file extensions, enforces a bounded read size, requires valid UTF-8, and
  walks the tree in a deterministic order. It performs no URL fetching and
  no shell execution.
- The MCP adapter is read-only and stdio-only: no ingestion, index rebuild,
  or evaluation tool is exposed, no automatic schema migration runs, and no
  filesystem scan happens at query time.
- The `sqlite-vec` SQLite extension is loaded from exactly one known,
  installed package location; extension loading is enabled only around
  that single load and disabled immediately after. There is no
  arbitrary or user-provided extension path.
- No credentials or secrets are required for local embeddings or any
  current capability.

## Limitations / non-goals

Deliberately out of scope for v0.1.0: answer generation, a chatbot UI, any
LLM in the retrieval path, cross-encoder or LLM reranking, document/source
filters, PDF or other non-text formats, remote source integrations (Drive,
Confluence, Notion, SharePoint, Slack, GitHub API), remote MCP transport, an
HTTP API, a cloud-hosted vector database, and autonomous or multi-agent
orchestration. This project is a retrieval and knowledge-serving plane, not
a RAG chatbot or an AI assistant.

## Development

```bash
pip install -e ".[dev,vector,local-embeddings,mcp]"

ruff check src tests
mypy src --strict
pytest -q
```

The unit suite does not download any model: tests exercising real
Sentence Transformers embeddings or `sqlite-vec` skip cleanly
(`pytest.importorskip`) when their optional dependency is absent, and CI
runs a base job with none of the optional extras installed to guarantee
that floor.
