"""
LangGraph Workflow
==================

Defines the StateGraph with conditional edges implementing the full
agent pipeline:

    START → Planner → Retriever →[confidence check]→ Web Search (optional)
          → Synthesizer → Verifier →[risk check]→ Refine & loop back
          → Output

Conditional edges:
1. After Retriever: if confidence < threshold → web search sub-step
2. After Verifier: if risk > threshold AND retries < max → refine query, loop back
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END

from src.config import get_settings
from src.graph.state import AgentState
from src.agents.planner import planner_node
from src.agents.retriever import retriever_node
from src.agents.synthesizer import synthesizer_node
from src.agents.verifier import verifier_node

logger = logging.getLogger(__name__)


# ── Conditional routing functions ────────────────────────────────────────────


def route_after_retriever(state: AgentState) -> str:
    """Decide whether to trigger web search based on retrieval confidence.

    Returns the name of the next node: 'web_search' or 'synthesizer'.
    """
    settings = get_settings()
    confidence = state.get("retrieval_confidence", 0.0)
    threshold = settings.retrieval_confidence_threshold

    if confidence < threshold:
        logger.info("Retrieval confidence %.2f < %.2f — triggering web search",
                     confidence, threshold)
        return "web_search"
    else:
        logger.info("Retrieval confidence %.2f >= %.2f — skipping web search",
                     confidence, threshold)
        return "synthesizer"


def route_after_verifier(state: AgentState) -> str:
    """Decide whether to loop back for refinement or output the answer.

    Returns 'refine' if risk is too high and retries remain, else 'output'.
    """
    settings = get_settings()
    risk = state.get("combined_risk_score", 0.0)
    threshold = state.get("risk_threshold", settings.risk_threshold)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_retries)

    if risk > threshold and retry_count < max_retries:
        logger.info("Risk %.2f > %.2f, retry %d/%d — refining query",
                     risk, threshold, retry_count, max_retries)
        return "refine"
    else:
        logger.info("Risk %.2f (threshold %.2f), retries %d/%d — outputting answer",
                     risk, threshold, retry_count, max_retries)
        return "output"


# ── Web search sub-step node ─────────────────────────────────────────────────


def web_search_node(state: AgentState) -> dict:
    """Sub-step: fetch web results when local retrieval confidence is low.

    Calls the Tavily search service for each search term (up to 3 to conserve
    quota), tags results as ``type: "web"``, deduplicates against any existing
    local chunks, and merges into ``web_results``.
    """
    from src.services import tavily_search

    logger.info("Web search sub-step triggered")

    search_terms = state.get("search_terms", [])
    local_chunks = state.get("local_chunks", [])

    # Fall back to the original / refined query if no search terms were planned
    if not search_terms:
        query = state.get("refined_query") or state.get("original_query", "")
        search_terms = [query]

    local_texts = {c.get("text", "")[:200].lower() for c in local_chunks}

    web_results: list[dict] = []

    for term in search_terms[:3]:  # limit to 3 terms to conserve Tavily quota
        try:
            results = tavily_search.search(query=term, max_results=3)
            for r in results:
                r["type"] = "web"
                # Deduplicate against local chunks
                snippet = r.get("text", "")[:200].lower()
                if any(snippet in lt or lt in snippet for lt in local_texts):
                    logger.debug("Skipping duplicate web result: %s…", snippet[:60])
                    continue
                web_results.append(r)
                local_texts.add(snippet)   # prevent intra-web duplicates too
        except Exception as e:
            logger.error("Tavily search failed for '%s': %s", term, e)

    logger.info("Web search node: %d results merged", len(web_results))

    return {
        "web_results": web_results,
        "used_web_search": True,
    }


# ── Refine node ──────────────────────────────────────────────────────────────


def refine_node(state: AgentState) -> dict:
    """Generate a refined query based on the verifier's feedback.

    Uses the original query + verification details to produce a more
    specific search query for the next retrieval attempt.
    """
    from groq import Groq

    settings = get_settings()
    retry_count = state.get("retry_count", 0)

    original = state["original_query"]
    draft = state.get("draft_answer", "")
    risk_label = state.get("risk_label", "UNKNOWN")
    details = state.get("verification_details", {})

    logger.info("Refining query (attempt %d)", retry_count + 1)

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.primary_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer. The previous answer to the "
                    "user's question was flagged as potentially unreliable "
                    f"(risk: {risk_label}). Rewrite the original question to be more "
                    "specific and targeted, adding constraints or context that would "
                    "help retrieve more authoritative information. "
                    "Return ONLY the refined query text, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {original}\n"
                    f"Previous answer (flagged): {draft[:500]}\n"
                    f"Refine this query for better retrieval."
                ),
            },
        ],
        temperature=0.3,
        max_tokens=256,
    )

    refined = response.choices[0].message.content.strip()
    logger.info("Refined query: %s", refined[:100])

    return {
        "refined_query": refined,
        "retry_count": retry_count + 1,
    }


# ── Output node ──────────────────────────────────────────────────────────────


def output_node(state: AgentState) -> dict:
    """Finalize the answer with its risk score for the user."""
    return {
        "final_answer": state.get("draft_answer", ""),
        "is_verified": True,
    }


# ── Graph Builder ────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct and compile the full LangGraph agent workflow.

    Returns a compiled graph ready to be invoked with an AgentState dict.
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("refine", refine_node)
    graph.add_node("output", output_node)

    # ── Add edges ──

    # START → Planner → Retriever
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")

    # Retriever → [conditional] → Web Search OR Synthesizer
    graph.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "web_search": "web_search",
            "synthesizer": "synthesizer",
        },
    )

    # Web Search → Synthesizer
    graph.add_edge("web_search", "synthesizer")

    # Synthesizer → Verifier
    graph.add_edge("synthesizer", "verifier")

    # Verifier → [conditional] → Refine OR Output
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "refine": "refine",
            "output": "output",
        },
    )

    # Refine → back to Retriever (loop)
    graph.add_edge("refine", "retriever")

    # Output → END
    graph.add_edge("output", END)

    return graph.compile()
