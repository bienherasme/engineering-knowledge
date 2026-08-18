from pathlib import Path

import pytest

from engineering_knowledge.config import ConfigurationError, load_config


def _write_config(config_dir: Path, body: str) -> Path:
    (config_dir / "corpus").mkdir(exist_ok=True)
    config_path = config_dir / "engineering-knowledge.toml"
    config_path.write_text(body)
    return config_path


def test_load_config_resolves_paths_relative_to_config_file_directory(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        [source]
        source_id = "demo"
        root = "./corpus"

        [persistence]
        db_path = "./data/ek.db"
        """,
    )

    config = load_config(config_path)

    assert config.source.root == (tmp_path / "corpus").resolve()
    assert config.persistence.db_path == (tmp_path / "data" / "ek.db").resolve()
    assert config.processing.max_chunk_chars == 1000
    assert config.retrieval.default_strategy.value == "lexical"
    assert config.embeddings.enabled is False


@pytest.mark.parametrize(
    "scenario",
    ["missing_file", "invalid_toml", "unsupported_embedding_provider"],
)
def test_load_config_raises_configuration_error(tmp_path: Path, scenario: str) -> None:
    if scenario == "missing_file":
        config_path = tmp_path / "does-not-exist.toml"
    elif scenario == "invalid_toml":
        config_path = tmp_path / "bad.toml"
        config_path.write_text("this is not [ valid toml")
    else:
        config_path = _write_config(
            tmp_path,
            """
            [source]
            source_id = "demo"
            root = "./corpus"

            [persistence]
            db_path = "./ek.db"

            [embeddings]
            enabled = true
            provider = "openai"
            """,
        )

    with pytest.raises(ConfigurationError):
        load_config(config_path)
