"""
Tavily Web Search Service
=========================

Wraps the Tavily Python client for agent-friendly web search results.
"""

from __future__ import annotations

import logging

from tavily import TavilyClient

from src.config import get_settings

logger = logging.getLogger(__name__)

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    """Lazy-initialise the Tavily client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = TavilyClient(api_key=settings.tavily_api_key)
        logger.info("Tavily client initialised")
    return _client


def search(query: str, max_results: int = 5) -> list[dict]:
    """Run a Tavily web search and return structured results.

    Parameters
    ----------
    query : str
        The search query.
    max_results : int
        Maximum number of results to return.

    Returns
    -------
    list[dict]
        Each dict has keys: ``title``, ``url``, ``text``, ``score``.
    """
    client = _get_client()

    logger.info("Tavily searching: %s (max %d)", query[:60], max_results)

    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_answer=False,
    )

    results = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "text": item.get("content", ""),
            "score": item.get("score", 0.0),
        })

    logger.info("Tavily returned %d results", len(results))
    return results
