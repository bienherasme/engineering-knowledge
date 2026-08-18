import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "corpus"


def test_mcp_module_is_optional_and_never_imported_by_base_code() -> None:
    """Base package, retrieval, and CLI parser construction never import 'mcp'.

    Run in a fresh subprocess so this proves independence regardless of
    whether an earlier test in the same process already imported it.
    """
    script = """
import sys
import engineering_knowledge
import engineering_knowledge.retrieval
from engineering_knowledge.cli.main import _build_parser

_build_parser().parse_args(["ingest", "--config", "unused.toml"])
assert "mcp" not in sys.modules, sorted(m for m in sys.modules if m.startswith("mcp"))
print("OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "OK"


def test_mcp_server_exposes_read_only_tools_with_structured_output_and_lookup_semantics(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mcp")
    import anyio
    from mcp.client.session import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    from engineering_knowledge.cli.main import main as cli_main
    from engineering_knowledge.composition import build_knowledge_service, open_repository
    from engineering_knowledge.config import load_config
    from engineering_knowledge.mcp.server import build_server

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
    assert cli_main(["ingest", "--config", str(config_path)]) == 0

    async def scenario() -> dict[str, Any]:
        config = load_config(config_path)
        repository = open_repository(config, read_only=True)
        knowledge_service = build_knowledge_service(config, repository)
        server = build_server(knowledge_service)

        outcome: dict[str, Any] = {}
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            server_read, server_write = server_streams
            client_read, client_write = client_streams

            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    server._lowlevel_server.run,
                    server_read,
                    server_write,
                    server._lowlevel_server.create_initialization_options(),
                )

                async with ClientSession(client_read, client_write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    outcome["tool_names"] = sorted(tool.name for tool in tools.tools)

                    search_result = await session.call_tool(
                        "search_knowledge", {"query": "MAX_RETRY_COUNT", "max_results": 3}
                    )
                    outcome["search_is_error"] = search_result.is_error
                    outcome["search_content"] = search_result.structured_content

                    missing = await session.call_tool(
                        "get_chunk", {"chunk_id": "chunk_does_not_exist"}
                    )
                    outcome["missing_found"] = missing.structured_content["found"]

                    invalid = await session.call_tool("search_knowledge", {"query": ""})
                    outcome["invalid_is_error"] = invalid.is_error
                    outcome["invalid_text"] = invalid.content[0].text

                tg.cancel_scope.cancel()

        repository.close()
        return outcome

    outcome = anyio.run(scenario)

    assert outcome["tool_names"] == ["get_chunk", "get_document", "search_knowledge"]
    assert outcome["search_is_error"] is False
    assert outcome["search_content"]["strategy"] == "lexical"
    assert outcome["search_content"]["results"]
    assert outcome["missing_found"] is False
    assert outcome["invalid_is_error"] is True
    assert "Traceback" not in outcome["invalid_text"]
