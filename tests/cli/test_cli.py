import json
from pathlib import Path

import pytest

from engineering_knowledge.cli.main import main

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "corpus"


def _write_lexical_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "engineering-knowledge.toml"
    config_path.write_text(
        f"""
        [source]
        source_id = "demo"
        root = "{CORPUS_DIR.as_posix()}"

        [persistence]
        db_path = "{(tmp_path / "ek.db").as_posix()}"

        [retrieval]
        default_strategy = "lexical"

        [embeddings]
        enabled = false
        """
    )
    return config_path


def test_search_dispatches_and_emits_provenance_without_generation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_lexical_config(tmp_path)
    assert main(["ingest", "--config", str(config_path)]) == 0
    capsys.readouterr()

    exit_code = main(["search", "MAX_RETRY_COUNT", "--config", str(config_path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["strategy"] == "lexical"
    assert payload["results"], "expected at least one hit"
    first_hit = payload["results"][0]
    assert first_hit["source_reference"]["relative_path"].endswith(".md")
    assert first_hit["chunk"]["chunk_id"].startswith("chunk_")
    assert first_hit["chunk"]["text"]
    assert "answer" not in payload


def test_rebuild_vectors_uses_the_configured_embedding_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sqlite_vec")
    from engineering_knowledge.embeddings import FakeEmbeddingProvider

    config_path = _write_lexical_config(tmp_path)
    assert main(["ingest", "--config", str(config_path)]) == 0
    capsys.readouterr()

    monkeypatch.setattr(
        "engineering_knowledge.cli.main.build_embedding_provider",
        lambda config: FakeEmbeddingProvider(dimension=4),
    )

    exit_code = main(["rebuild-vectors", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "provider: fake" in captured.out
    assert "dimension: 4" in captured.out
    assert "reindexed chunks:" in captured.out


def test_expected_error_returns_nonzero_exit_and_concise_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_config = tmp_path / "nope.toml"

    exit_code = main(["search", "anything", "--config", str(missing_config)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


def test_json_mode_emits_parseable_ingestion_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_lexical_config(tmp_path)

    exit_code = main(["ingest", "--config", str(config_path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["source_id"] == "demo"
    assert payload["discovered"] == 5
    assert payload["created"] == 5
