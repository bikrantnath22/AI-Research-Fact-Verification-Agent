"""
Synthesis Agent
===============

Drafts an answer from retrieved chunks (local + web) with inline
source attribution using ``[Source N]`` markers.
"""

from __future__ import annotations

import json
import logging

from groq import Groq

from src.config import get_settings
from src.graph.state import AgentState

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """\
You are a research synthesis assistant. Given a user's question and a set of
retrieved source documents, write a clear, accurate answer.

Rules:
1. Use ONLY information from the provided sources.  Do NOT invent facts.
2. Cite sources inline using [Source N] notation (e.g., "According to [Source 1], ...").
3. If multiple sources agree, cite them together: [Source 1, Source 3].
4. If the sources are insufficient to fully answer the question, explicitly say so.
5. Keep the answer concise but thorough (3-8 sentences).
6. End with a brief "Sources Used" section listing which source numbers you cited.
7. CRITICAL: DO NOT use or call any tools (like browser.run). Output plain text only.
"""


def _format_sources_context(state: AgentState) -> tuple[str, list[dict]]:
    """Build a numbered source context string and a source index list.

    Returns
    -------
    tuple[str, list[dict]]
        - Formatted context string with numbered sources.
        - Source index: list of dicts mapping Source N → origin info.
    """
    sources: list[dict] = []
    context_parts: list[str] = []
    idx = 1

    # Local chunks first
    for chunk in state.get("local_chunks", []):
        label = f"[Source {idx}]"
        source_info = {
            "id": idx,
            "type": "local",
            "text": chunk["text"][:200],
            "ref": chunk.get("source", "uploaded_document"),
            "score": chunk.get("score", 0.0),
        }
        sources.append(source_info)
        context_parts.append(f"{label} (Local document: {source_info['ref']})\n{chunk['text']}")
        idx += 1

    # Web results
    for result in state.get("web_results", []):
        label = f"[Source {idx}]"
        source_info = {
            "id": idx,
            "type": "web",
            "text": result["text"][:200],
            "ref": result.get("url", result.get("title", "web")),
            "title": result.get("title", ""),
            "url": result.get("url", ""),
        }
        sources.append(source_info)
        context_parts.append(
            f"{label} (Web: {source_info['title']} — {source_info['url']})\n{result['text']}"
        )
        idx += 1

    context = "\n\n---\n\n".join(context_parts) if context_parts else "(No sources available)"
    return context, sources


def synthesizer_node(state: AgentState) -> dict:
    """LangGraph node: synthesize an answer from retrieved context.

    Calls Groq with the retrieved chunks formatted as numbered sources,
    instructing the LLM to produce an answer with inline ``[Source N]``
    citations.
    """
    settings = get_settings()
    query = state.get("refined_query") or state["original_query"]

    context, sources = _format_sources_context(state)

    n_local = len(state.get("local_chunks", []))
    n_web = len(state.get("web_results", []))
    logger.info("Synthesizer: %d local + %d web sources for query: %s",
                n_local, n_web, query[:80])

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.primary_model,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Retrieved Sources:\n\n{context}\n\n"
                    f"Please answer the question using the sources above."
                ),
            },
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    draft = response.choices[0].message.content.strip()
    logger.info("Synthesizer produced answer (%d chars)", len(draft))

    return {
        "draft_answer": draft,
        "sources": sources,
    }
