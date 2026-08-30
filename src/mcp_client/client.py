"""
MCP Client
==========

Generic async MCP client for connecting to the Qdrant and Tavily MCP
servers.  Supports Streamable HTTP transport (Docker/k3s) and stdio
(local dev), gated by ``MCP_TRANSPORT`` env var.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.config import get_settings

logger = logging.getLogger(__name__)


async def call_tool(
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """Call an MCP tool on a remote server via Streamable HTTP transport.

    Parameters
    ----------
    server_url : str
        The MCP server URL (e.g., ``http://mcp-qdrant:8001/mcp``).
    tool_name : str
        The name of the tool to invoke.
    arguments : dict
        Arguments to pass to the tool.

    Returns
    -------
    Any
        The parsed result from the MCP tool (typically a dict or list).
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    logger.info("MCP call: %s → %s(%s)", server_url, tool_name, list(arguments.keys()))

    async with streamablehttp_client(server_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

            # Extract text content from the MCP result
            if result.content:
                text = result.content[0].text
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, AttributeError):
                    return text

            return None


async def call_qdrant_search(
    query: str,
    collection: str,
    top_k: int = 5,
) -> list[dict]:
    """Convenience: search Qdrant via MCP.

    Parameters
    ----------
    query : str
        Search query text.
    collection : str
        Qdrant collection name.
    top_k : int
        Number of results.

    Returns
    -------
    list[dict]
        Search results from the Qdrant MCP server.
    """
    settings = get_settings()
    result = await call_tool(
        server_url=settings.mcp_qdrant_url,
        tool_name="qdrant_search",
        arguments={"query": query, "collection": collection, "top_k": top_k},
    )
    return result if isinstance(result, list) else []


async def call_qdrant_upsert(
    texts: list[str],
    collection: str,
    metadata: Optional[list[dict]] = None,
) -> dict:
    """Convenience: upsert into Qdrant via MCP."""
    settings = get_settings()
    args: dict = {"texts": texts, "collection": collection}
    if metadata:
        args["metadata"] = metadata

    result = await call_tool(
        server_url=settings.mcp_qdrant_url,
        tool_name="qdrant_upsert",
        arguments=args,
    )
    return result if isinstance(result, dict) else {"upserted": 0}


async def call_web_search(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """Convenience: search the web via Tavily MCP.

    Parameters
    ----------
    query : str
        Search query.
    max_results : int
        Max results.

    Returns
    -------
    list[dict]
        Web search results from the Tavily MCP server.
    """
    settings = get_settings()
    result = await call_tool(
        server_url=settings.mcp_tavily_url,
        tool_name="web_search",
        arguments={"query": query, "max_results": max_results},
    )
    return result if isinstance(result, list) else []
