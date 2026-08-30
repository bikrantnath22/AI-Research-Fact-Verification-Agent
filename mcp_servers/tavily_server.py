"""
MCP Server — Tavily Web Search
================================

Exposes Tavily web search as an MCP tool via Streamable HTTP transport
(for Docker/k3s) or stdio (for local dev).

Tools:
- ``web_search``: Search the web using Tavily API.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP

from src.services.tavily_search import search as tavily_search

# ── MCP Server Setup ─────────────────────────────────────────────────────────

mcp = FastMCP(
    "Tavily Web Search Server",
    description="MCP server for agent-friendly web search via the Tavily API",
)


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def web_search(
    query: str,
    max_results: int = 5,
) -> str:
    """Search the web using the Tavily API.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 5).

    Returns:
        JSON string of search results, each with 'title', 'url', 'text', 'score'.
    """
    results = tavily_search(query=query, max_results=max_results)
    return json.dumps(results, default=str)


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tavily MCP Server")
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default=os.getenv("MCP_TRANSPORT", "http"),
        help="MCP transport: 'http' (Streamable HTTP) or 'stdio'",
    )
    parser.add_argument("--port", type=int, default=8002, help="HTTP port (default: 8002)")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = args.port
        mcp.run(transport="sse")
