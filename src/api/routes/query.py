"""
Query Route
===========

POST /query — runs the full LangGraph agent pipeline and returns the
verified answer with sources and risk score.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from src.api.models import QueryRequest, QueryResponse, SourceInfo, VerificationDetails
from src.config import get_settings
from src.graph.workflow import build_graph

logger = logging.getLogger(__name__)

router = APIRouter()

# Build the graph once at module level
_graph = None


def _get_graph():
    """Lazy-build the compiled LangGraph workflow."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@router.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    """Run the full research agent pipeline on the user's question.

    The pipeline:
    1. **Planner** — decomposes the query into sub-questions + search terms.
    2. **Retriever** — searches Qdrant (local docs) + conditional Tavily (web).
    3. **Synthesizer** — drafts an answer with inline source attribution.
    4. **Verifier** — semantic entropy + ensemble disagreement → risk score.
    5. **Critique loop** — if risk too high, refine query and re-retrieve (max N retries).
    """
    settings = get_settings()
    graph = _get_graph()

    logger.info("Query received: %s (collection: %s)",
                req.question[:80], req.collection or "none")

    # Build initial state
    initial_state = {
        "original_query": req.question,
        "uploaded_doc_collection": req.collection,
        "retry_count": 0,
        "max_retries": settings.max_retries,
        "risk_threshold": settings.risk_threshold,
    }

    try:
        # Run the graph (synchronous nodes — run in thread pool)
        result = await asyncio.to_thread(graph.invoke, initial_state)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Build response
    sources = [
        SourceInfo(**s) if isinstance(s, dict) else s
        for s in result.get("sources", [])
    ]

    verification_details = result.get("verification_details", {})

    return QueryResponse(
        question=req.question,
        answer=result.get("final_answer", result.get("draft_answer", "")),
        sources=sources,
        used_web_search=result.get("used_web_search", False),
        retrieval_confidence=round(result.get("retrieval_confidence", 0.0), 4),
        combined_risk_score=round(result.get("combined_risk_score", 0.0), 4),
        risk_label=result.get("risk_label", "LOW"),
        semantic_entropy_score=round(result.get("semantic_entropy_score", 0.0), 4),
        ensemble_disagreement_score=round(result.get("ensemble_disagreement_score", 0.0), 4),
        verification_details=VerificationDetails(**verification_details) if verification_details else VerificationDetails(),
        retry_count=result.get("retry_count", 0),
        refined_query=result.get("refined_query", ""),
    )
