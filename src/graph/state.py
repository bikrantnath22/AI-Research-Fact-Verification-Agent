"""
LangGraph State Schema
======================

Defines the TypedDict that flows through every node in the agent graph.
Each agent reads from and writes to specific fields in this state.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state flowing through the LangGraph agent pipeline.

    Fields are grouped by the agent that primarily writes to them.
    ``total=False`` allows nodes to only update the fields they own.
    """

    # ── Input ─────────────────────────────────────────────────────────────
    original_query: str
    uploaded_doc_collection: str  # Qdrant collection name (empty if no docs)

    # ── Planner Output ────────────────────────────────────────────────────
    sub_questions: list[str]
    search_terms: list[str]

    # ── Retriever Output ──────────────────────────────────────────────────
    local_chunks: list[dict]       # [{text, source, score}, ...]
    web_results: list[dict]        # [{text, url, title}, ...]
    retrieval_confidence: float    # 0–1 average similarity score
    used_web_search: bool

    # ── Synthesizer Output ────────────────────────────────────────────────
    draft_answer: str
    sources: list[dict]            # [{text, type: "local"|"web", ref}, ...]

    # ── Verifier Output ───────────────────────────────────────────────────
    semantic_entropy_score: float
    ensemble_disagreement_score: float
    combined_risk_score: float
    risk_label: str                # "LOW" | "MEDIUM" | "HIGH"
    verification_details: dict

    # ── Critique / Refine Loop ────────────────────────────────────────────
    refined_query: str
    retry_count: int
    max_retries: int               # default 2
    risk_threshold: float          # default 0.6
    final_answer: str
    is_verified: bool
