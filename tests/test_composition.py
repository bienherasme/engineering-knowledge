import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "corpus"


def test_lexical_composition_never_imports_optional_embedding_dependency(tmp_path: Path) -> None:
    """A lexical-only build never loads sentence-transformers, even if installed.

    Run in a fresh subprocess so the check is not contaminated by an
    earlier test in the same suite having already imported it.
    """
    config_path = tmp_path / "engineering-knowledge.toml"
    config_path.write_text(
        f"""
        [source]
        source_id = "demo"
        root = "{CORPUS_DIR.as_posix()}"

        [persistence]
        db_path = "{(tmp_path / "ek.db").as_posix()}"

        [embeddings]
        enabled = false
        """
    )

    script = f"""
import sys
from pathlib import Path
from engineering_knowledge.config import load_config
from engineering_knowledge.composition import (
    build_ingestion_service, build_knowledge_service, build_source_adapter, open_repository,
)

config = load_config(Path({str(config_path)!r}))
repository = open_repository(config)
adapter = build_source_adapter(config)
service = build_ingestion_service(config, repository)
service.ingest_source(adapter)
knowledge_service = build_knowledge_service(config, repository)
result = knowledge_service.search("MAX_RETRY_COUNT")
assert result.results, "expected at least one lexical hit"
repository.close()
assert "sentence_transformers" not in sys.modules
print("OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "OK"
