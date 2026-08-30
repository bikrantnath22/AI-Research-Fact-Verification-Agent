"""
MCP Server — Qdrant Retrieval
==============================

Exposes Qdrant vector search and upsert as MCP tools via Streamable HTTP
transport (for Docker/k3s) or stdio (for local dev).

Tools:
- ``qdrant_search``: Embed a query and search a collection.
- ``qdrant_upsert``: Upsert pre-embedded chunks into a collection.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the project root is on sys.path so we can import src.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP

from src.config import get_settings
from src.services.embeddings import embed_text, embed_batch
from src.services.qdrant import (
    create_collection,
    upsert_chunks,
    search,
    collection_exists,
)

# ── MCP Server Setup ─────────────────────────────────────────────────────────

mcp = FastMCP(
    "Qdrant Retrieval Server",
    description="MCP server for vector database retrieval and storage via Qdrant",
)


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def qdrant_search(
    query: str,
    collection: str,
    top_k: int = 5,
) -> str:
    """Search a Qdrant collection for chunks similar to the query.

    Args:
        query: The search query text (will be embedded automatically).
        collection: The Qdrant collection name to search.
        top_k: Number of top results to return (default: 5).

    Returns:
        JSON string of search results, each with 'text', 'score', 'source'.
    """
    if not collection_exists(collection):
        return json.dumps({"error": f"Collection '{collection}' does not exist", "results": []})

    query_vector = embed_text(query)
    results = search(collection=collection, query_vector=query_vector, top_k=top_k)
    return json.dumps(results, default=str)


@mcp.tool()
def qdrant_upsert(
    texts: list[str],
    collection: str,
    metadata: list[dict] | None = None,
) -> str:
    """Embed and upsert text chunks into a Qdrant collection.

    Creates the collection if it doesn't exist.

    Args:
        texts: List of text chunks to embed and store.
        collection: Target Qdrant collection name.
        metadata: Optional list of metadata dicts, one per text chunk.

    Returns:
        JSON string with the number of chunks upserted.
    """
    create_collection(collection)
    vectors = embed_batch(texts)
    n = upsert_chunks(collection=collection, texts=texts, vectors=vectors, metadata=metadata)
    return json.dumps({"upserted": n, "collection": collection})


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qdrant MCP Server")
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default=os.getenv("MCP_TRANSPORT", "http"),
        help="MCP transport: 'http' (Streamable HTTP) or 'stdio'",
    )
    parser.add_argument("--port", type=int, default=8001, help="HTTP port (default: 8001)")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = args.port
        mcp.run(transport="sse")
