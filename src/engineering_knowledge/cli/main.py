"""Argparse-based CLI entry point.

Five commands, each a thin adapter: ``ingest``, ``search``,
``rebuild-vectors``, ``evaluate``, ``serve-mcp``. Every command composes
already-built services through ``engineering_knowledge.composition``
rather than talking to persistence, retrieval, or evaluation internals
directly.

``main`` catches exactly the expected application-boundary error types and
turns each into a concise stderr message and a nonzero exit code. Anything
else is a programming bug and is left to propagate as a traceback rather
than being folded into "just another CLI error".
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from engineering_knowledge.composition import (
    build_embedding_provider,
    build_ingestion_service,
    build_knowledge_service,
    build_source_adapter,
    build_vector_retriever,
    open_repository,
)
from engineering_knowledge.config import ConfigurationError, load_config
from engineering_knowledge.embeddings.base import EmbeddingError
from engineering_knowledge.evaluation import (
    DEFAULT_K,
    EvaluationDatasetError,
    EvaluationReport,
    EvaluationRunnerError,
    GoldenDataset,
    run_evaluation,
)
from engineering_knowledge.ingestion.service import IngestionResult
from engineering_knowledge.persistence.base import PersistenceError
from engineering_knowledge.retrieval.errors import (
    IndexUnavailableError,
    InvalidQueryError,
    VectorIndexIncompatibleError,
    VectorIndexUnavailableError,
)
from engineering_knowledge.retrieval.service import (
    DEFAULT_PUBLIC_MAX_RESULTS,
    RetrievalResult,
    RetrievalStrategy,
)
from engineering_knowledge.sources.base import SourceConfigurationError, SourceReadError

DEFAULT_CONFIG_PATH = Path("engineering-knowledge.toml")

_EXPECTED_ERRORS = (
    ConfigurationError,
    InvalidQueryError,
    SourceConfigurationError,
    SourceReadError,
    PersistenceError,
    IndexUnavailableError,
    VectorIndexUnavailableError,
    VectorIndexIncompatibleError,
    EmbeddingError,
    EvaluationDatasetError,
    EvaluationRunnerError,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)  # type: ignore[no-any-return]
    except _EXPECTED_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engineering-knowledge",
        description="Local-first retrieval over a prepared engineering knowledge base.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest the configured source into the local knowledge database."
    )
    _add_config_argument(ingest_parser)
    ingest_parser.add_argument(
        "--json", action="store_true", help="Emit a JSON ingestion summary instead of text."
    )
    ingest_parser.set_defaults(handler=_cmd_ingest)

    search_parser = subparsers.add_parser(
        "search", help="Search the local knowledge database and print retrieval evidence."
    )
    _add_config_argument(search_parser)
    search_parser.add_argument("query", help="Free-text search query.")
    search_parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in RetrievalStrategy],
        default=None,
        help="Retrieval strategy. Defaults to the configured retrieval.default_strategy.",
    )
    search_parser.add_argument(
        "--max-results", type=int, default=DEFAULT_PUBLIC_MAX_RESULTS, dest="max_results"
    )
    search_parser.add_argument(
        "--json", action="store_true", help="Emit the JSON RetrievalResult instead of text."
    )
    search_parser.set_defaults(handler=_cmd_search)

    rebuild_parser = subparsers.add_parser(
        "rebuild-vectors",
        help="Rebuild the vector index from persisted chunks using the configured embedding "
        "provider.",
    )
    _add_config_argument(rebuild_parser)
    rebuild_parser.set_defaults(handler=_cmd_rebuild_vectors)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run the golden-dataset evaluation across lexical, vector, and hybrid retrieval.",
    )
    _add_config_argument(evaluate_parser)
    evaluate_parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to a golden_dataset.json file."
    )
    evaluate_parser.add_argument("--k", type=int, default=DEFAULT_K)
    evaluate_parser.add_argument(
        "--json", action="store_true", help="Emit the JSON EvaluationReport instead of text."
    )
    evaluate_parser.set_defaults(handler=_cmd_evaluate)

    serve_parser = subparsers.add_parser(
        "serve-mcp", help="Serve the local knowledge database over MCP stdio, read-only."
    )
    _add_config_argument(serve_parser)
    serve_parser.set_defaults(handler=_cmd_serve_mcp)

    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to a TOML configuration file (default: {DEFAULT_CONFIG_PATH}).",
    )


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    embedding_provider = build_embedding_provider(config) if config.embeddings.enabled else None
    repository = open_repository(config, vector_index_enabled=config.embeddings.enabled)
    try:
        adapter = build_source_adapter(config)
        service = build_ingestion_service(config, repository, embedding_provider=embedding_provider)
        result = service.ingest_source(adapter)
    finally:
        repository.close()

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_ingestion_result(result)
    return 0


def _cmd_rebuild_vectors(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    embedding_provider = build_embedding_provider(config)
    repository = open_repository(config, vector_index_enabled=True)
    try:
        summary = repository.rebuild_vector_index(embedding_provider)
    finally:
        repository.close()

    print(f"provider: {embedding_provider.provider_type}")
    print(f"model: {embedding_provider.model_id}")
    print(f"dimension: {embedding_provider.dimension}")
    print(f"embedding fingerprint: {summary.embedding_fingerprint}")
    print(f"reindexed chunks: {summary.reindexed}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    explicit_strategy = RetrievalStrategy(args.strategy) if args.strategy else None
    effective_strategy = explicit_strategy or config.retrieval.default_strategy
    needs_vector = effective_strategy is not RetrievalStrategy.LEXICAL

    embedding_provider = build_embedding_provider(config) if needs_vector else None
    repository = open_repository(config, vector_index_enabled=needs_vector)
    try:
        vector_retriever = (
            build_vector_retriever(embedding_provider, repository)
            if embedding_provider is not None
            else None
        )
        knowledge_service = build_knowledge_service(
            config, repository, vector_retriever=vector_retriever
        )
        result = knowledge_service.search(
            args.query, strategy=explicit_strategy, max_results=args.max_results
        )
    finally:
        repository.close()

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_search_result(result)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    # Evaluation always compares lexical, vector, and hybrid, so it always
    # needs real vector capability; there is no partial/lexical-only mode
    # to silently fall back to.
    embedding_provider = build_embedding_provider(config)
    repository = open_repository(config, vector_index_enabled=True)
    try:
        vector_retriever = build_vector_retriever(embedding_provider, repository)
        knowledge_service = build_knowledge_service(
            config, repository, vector_retriever=vector_retriever
        )
        dataset = GoldenDataset.load(args.dataset)
        report = run_evaluation(knowledge_service, dataset, k=args.k)
    finally:
        repository.close()

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_evaluation_report(report)
    return 0


def _cmd_serve_mcp(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        from engineering_knowledge.mcp.server import run_stdio_server
    except ImportError as error:
        raise ConfigurationError(
            "serve-mcp requires the optional 'mcp' dependency; install the 'mcp' extra"
        ) from error
    run_stdio_server(config)
    return 0


def _print_ingestion_result(result: IngestionResult) -> None:
    print(f"source: {result.source_id}")
    print(f"discovered: {result.discovered}")
    print(f"created: {result.created}")
    print(f"updated: {result.updated}")
    print(f"reprocessed: {result.reprocessed}")
    print(f"unchanged: {result.unchanged}")
    print(f"deleted: {result.deleted}")
    if result.vector_indexing is not None:
        print(f"vector reindexed: {result.vector_indexing.reindexed}")
        print(f"embedding fingerprint: {result.vector_indexing.embedding_fingerprint}")


def _print_search_result(result: RetrievalResult) -> None:
    print(f"strategy: {result.strategy.value}  status: {result.status.value}")
    if result.truncated:
        print(f"truncated: yes ({result.truncation_reason})")
    if not result.results:
        print("(no results)")
        return

    for hit in result.results:
        reference = hit.source_reference
        section = " > ".join(reference.section_path.headings) or "(root)"
        print(f"\n[{hit.rank}] {reference.relative_path}  section: {section}")
        print(f"    chunk_id: {reference.chunk_id}")

        diagnostics = []
        if hit.bm25_score is not None:
            diagnostics.append(f"bm25={hit.bm25_score:.4f}")
        if hit.vector_distance is not None:
            diagnostics.append(f"distance={hit.vector_distance:.4f}")
        if hit.rrf_score is not None:
            diagnostics.append(f"rrf={hit.rrf_score:.4f}")
        if diagnostics:
            print(f"    {'  '.join(diagnostics)}")

        excerpt = " ".join(hit.chunk.text.split())
        if len(excerpt) > 200:
            excerpt = excerpt[:200] + "..."
        print(f"    {excerpt}")


def _print_evaluation_report(report: EvaluationReport) -> None:
    print(f"queries: {report.total_queries}  k: {report.k}\n")
    print(f"{'strategy':<10} {'recall@' + str(report.k):<12} {'mrr':<8}")
    for strategy_eval in report.strategies:
        print(
            f"{strategy_eval.strategy.value:<10} "
            f"{strategy_eval.mean_recall_at_k:<12.4f} "
            f"{strategy_eval.mrr:<8.4f}"
        )

    for strategy_eval in report.strategies:
        print(f"\n{strategy_eval.strategy.value} by category:")
        for breakdown in strategy_eval.category_breakdown:
            print(
                f"  {breakdown.category.value:<10} "
                f"recall@{report.k}={breakdown.mean_recall_at_k:.4f} "
                f"mrr={breakdown.mrr:.4f} (n={breakdown.query_count})"
            )
