"""Read-only MCP stdio adapter over an already-prepared knowledge database.

Deliberately empty at package level: nothing here imports ``server``, so
importing this package never requires the optional ``mcp`` SDK. Only
``engineering_knowledge.mcp.server`` itself, loaded lazily by the CLI's
``serve-mcp`` command, imports it.
"""
