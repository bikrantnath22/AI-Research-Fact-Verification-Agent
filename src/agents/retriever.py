"""
Retriever Agent
===============

Embeds search terms, queries Qdrant for local document chunks, computes a
retrieval confidence score, and conditionally triggers Tavily web search
when local coverage is weak.  Merges and deduplicates results.

In Phase 6 the direct service calls will be replaced by MCP client calls,
but the logic and interface remain identical.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import get_settings
from src.graph.state import AgentState
from src.services import embeddings
from src.services import qdrant as qdrant_service
from src.services import tavily_search

logger = logging.getLogger(__name__)


def _deduplicate_chunks(
    local: list[dict],
    web: list[dict],
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """Merge local and web results, removing near-duplicate texts.

    Simple deduplication by checking if any web result text is a substring
    of a local chunk (or vice-versa).  A more robust approach would use
    embedding cosine similarity, but this is sufficient for the demo.
    """
    merged = list(local)  # local results take priority
    local_texts = {c["text"][:200].lower() for c in local}

    for web_chunk in web:
        short = web_chunk["text"][:200].lower()
        # Skip if substantially overlapping with an existing local chunk
        if any(short in lt or lt in short for lt in local_texts):
            logger.debug("Skipping duplicate web result: %s…", short[:60])
            continue
        merged.append(web_chunk)

    return merged


def retriever_node(state: AgentState) -> dict:
    """LangGraph node: retrieve relevant chunks from Qdrant + conditional web search.

    Steps:
    1. Embed each search term.
    2. Query Qdrant for each term (if a collection exists).
    3. Compute retrieval confidence = average of top-k similarity scores.
    4. If confidence < threshold: call Tavily, merge + deduplicate.
    """
    settings = get_settings()
    search_terms = state.get("search_terms", [])
    collection = state.get("uploaded_doc_collection", "")

    if not search_terms:
        query = state.get("refined_query") or state.get("original_query", "")
        search_terms = [query]

    logger.info("Retriever processing %d search terms", len(search_terms))

    # ── Step 1-2: Local retrieval from Qdrant ────────────────────────────
    local_chunks: list[dict] = []
    all_scores: list[float] = []

    if collection and qdrant_service.collection_exists(collection):
        for term in search_terms:
            query_vector = embeddings.embed_text(term)
            results = qdrant_service.search(
                collection=collection,
                query_vector=query_vector,
                top_k=5,
            )
            for chunk in results:
                chunk["type"] = "local"
                local_chunks.append(chunk)
                all_scores.append(chunk.get("score", 0.0))

        logger.info("Local retrieval: %d chunks from '%s'",
                     len(local_chunks), collection)
    else:
        logger.info("No document collection — skipping local retrieval")

    # ── Step 3: Compute retrieval confidence ─────────────────────────────
    if all_scores:
        retrieval_confidence = sum(all_scores) / len(all_scores)
    else:
        retrieval_confidence = 0.0

    logger.info("Retrieval confidence: %.3f (threshold: %.3f)",
                retrieval_confidence, settings.retrieval_confidence_threshold)

    # ── Step 4: Conditional web search ───────────────────────────────────
    web_results: list[dict] = []
    used_web_search = False

    if retrieval_confidence < settings.retrieval_confidence_threshold:
        logger.info("Low confidence — triggering Tavily web search")
        used_web_search = True

        for term in search_terms[:3]:  # Limit to 3 terms to conserve Tavily quota
            try:
                results = tavily_search.search(query=term, max_results=3)
                for r in results:
                    r["type"] = "web"
                web_results.extend(results)
            except Exception as e:
                logger.error("Tavily search failed for '%s': %s", term, e)

        logger.info("Web search returned %d results", len(web_results))

    # ── Merge & deduplicate ──────────────────────────────────────────────
    if web_results:
        all_chunks = _deduplicate_chunks(local_chunks, web_results)
    else:
        all_chunks = local_chunks

    return {
        "local_chunks": local_chunks,
        "web_results": web_results,
        "retrieval_confidence": retrieval_confidence,
        "used_web_search": used_web_search,
    }
