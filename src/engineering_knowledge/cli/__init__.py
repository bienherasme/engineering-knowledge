"""Command-line adapter over the existing application services.

Composes ``IngestionService``, ``KnowledgeService``, and the evaluation
runner from typed configuration; it never reimplements retrieval,
ingestion, or evaluation logic itself.
"""
