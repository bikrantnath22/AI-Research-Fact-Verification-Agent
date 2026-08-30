"""
Planner Agent
=============

Breaks the user's query into sub-questions and search terms
using a Groq LLM call with structured output.
"""

from __future__ import annotations

import json
import logging

from groq import Groq

from src.config import get_settings
from src.graph.state import AgentState

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are a research planning assistant. Given a user question, decompose it into
2-4 focused sub-questions and generate concise search terms for each.

Respond ONLY with valid JSON in this exact format:
{
  "sub_questions": ["...", "..."],
  "search_terms": ["...", "..."]
}

Rules:
- Sub-questions should cover different facets of the original query.
- Search terms should be short keyword phrases suitable for vector search.
- Do NOT include any text outside the JSON object.
- CRITICAL: DO NOT use or call any tools (like browser.run or web_search). Just output the raw JSON object.
"""


def planner_node(state: AgentState) -> dict:
    """LangGraph node: decompose query into sub-questions and search terms.

    On retry loops, uses ``refined_query`` instead of ``original_query``.
    """
    settings = get_settings()
    query = state.get("refined_query") or state["original_query"]

    logger.info("Planner processing: %s", query[:80])

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.primary_model,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    # Parse structured JSON response
    try:
        parsed = json.loads(raw)
        sub_questions = parsed.get("sub_questions", [query])
        search_terms = parsed.get("search_terms", [query])
    except (json.JSONDecodeError, KeyError):
        logger.warning("Planner returned non-JSON, falling back to original query")
        sub_questions = [query]
        search_terms = [query]

    logger.info("Planner produced %d sub-questions, %d search terms",
                len(sub_questions), len(search_terms))

    return {
        "sub_questions": sub_questions,
        "search_terms": search_terms,
    }
