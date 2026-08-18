"""Typed local TOML configuration.

One small config surface, not a settings framework: a TOML file describes
the source, database, processing, retrieval, and embedding configuration
for one local deployment, and ``load_config`` turns it into a validated,
frozen ``AppConfig`` that the CLI and MCP adapters compose services from.

Relative filesystem paths in the TOML file (``source.root``,
``persistence.db_path``) are resolved relative to the config file's own
directory, never the process's current working directory: a config file
should mean the same thing regardless of where it happens to be invoked
from. The resolved absolute paths live only in this runtime configuration
object; they never reach domain identity or a public retrieval result.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from engineering_knowledge.embeddings.sentence_transformers_provider import DEFAULT_MODEL_ID
from engineering_knowledge.retrieval import RetrievalStrategy
from engineering_knowledge.sources.local_filesystem import DEFAULT_MAX_FILE_SIZE_BYTES

_SUPPORTED_EMBEDDING_PROVIDERS = frozenset({"sentence_transformers"})


class ConfigurationError(Exception):
    """An expected configuration-boundary failure.

    Covers a missing config file, invalid TOML, a configuration that fails
    validation, an unsupported configured capability (such as an
    unrecognized embedding provider), or a capability required by explicit
    configuration whose optional dependency is not installed. Never used
    for ordinary retrieval/source/persistence failures, which already have
    their own typed errors.
    """


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    root: Path
    max_file_size_bytes: int = Field(gt=0)


class PersistenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    db_path: Path


class ProcessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_chunk_chars: int = Field(default=1000, gt=0)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_strategy: RetrievalStrategy = RetrievalStrategy.LEXICAL


class EmbeddingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    provider: str = "sentence_transformers"
    model_id: str = DEFAULT_MODEL_ID

    @model_validator(mode="after")
    def _check_provider(self) -> EmbeddingsConfig:
        if not self.enabled:
            return self
        if self.provider not in _SUPPORTED_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"unsupported embedding provider {self.provider!r}; "
                f"supported: {sorted(_SUPPORTED_EMBEDDING_PROVIDERS)}"
            )
        if not self.model_id.strip():
            raise ValueError("model_id must not be blank when embeddings are enabled")
        return self


class AppConfig(BaseModel):
    """Fully resolved, validated local configuration.

    Constructed only through ``load_config``: that is where TOML parsing
    and config-file-relative path resolution happen, so this model never
    needs to know where it was loaded from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: SourceConfig
    persistence: PersistenceConfig
    processing: ProcessingConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()


def load_config(config_path: Path) -> AppConfig:
    """Load, validate, and path-resolve one TOML configuration file.

    Filesystem existence of ``source.root`` or ``persistence.db_path`` is
    deliberately not checked here: that belongs to the adapters that
    actually open them (a writable repository may create its database
    file; a source adapter validates its root directory), not to a pure
    configuration model.
    """
    if not config_path.is_file():
        raise ConfigurationError(f"config file not found: {config_path}")

    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"invalid TOML in {config_path}: {error}") from error

    config_dir = config_path.resolve().parent

    try:
        source_raw = raw.get("source", {})
        persistence_raw = raw.get("persistence", {})
        return AppConfig(
            source=SourceConfig(
                source_id=source_raw.get("source_id", ""),
                root=_resolve_path(config_dir, source_raw.get("root", "")),
                max_file_size_bytes=source_raw.get(
                    "max_file_size_bytes", DEFAULT_MAX_FILE_SIZE_BYTES
                ),
            ),
            persistence=PersistenceConfig(
                db_path=_resolve_path(config_dir, persistence_raw.get("db_path", "")),
            ),
            processing=ProcessingConfig.model_validate(raw.get("processing", {})),
            retrieval=RetrievalConfig.model_validate(raw.get("retrieval", {})),
            embeddings=EmbeddingsConfig.model_validate(raw.get("embeddings", {})),
        )
    except (ValidationError, ValueError) as error:
        raise ConfigurationError(f"invalid configuration in {config_path}: {error}") from error


def _resolve_path(config_dir: Path, raw_value: str) -> Path:
    if not raw_value.strip():
        raise ValueError("path must not be blank")
    candidate = Path(raw_value)
    return candidate if candidate.is_absolute() else (config_dir / candidate).resolve()
