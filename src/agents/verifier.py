"""
Verifier Agent
==============

Runs hallucination detection on the draft answer using three complementary
methods:

1. **Semantic Entropy (50% weight):**
   Generate N samples at high temperature, cluster by bidirectional NLI
   entailment (HF Inference API), compute Shannon entropy → risk score.

   *Context-aware*: if the query references an uploaded document (contains
   keywords like "pdf", "document", "paper", etc.), the retrieved source
   chunks are prepended to the prompt so the LLM can answer from context
   instead of saying "I don't have the PDF".

2. **Ensemble Disagreement (30% weight):**
   Extract claims from answers by two different Groq models, check for
   NLI contradictions.

3. **Faithfulness (20% weight):**
   Extract factual claims from the synthesized answer, check each claim
   against the retrieved source chunks via NLI. Detects cases where the
   Synthesizer fabricated details not present in any source.

Combined: ``0.5 * entropy_risk + 0.3 * disagreement_score + 0.2 * faithfulness_score``
"""

from __future__ import annotations

import logging
import math
import re

from src.config import get_settings
from src.graph.state import AgentState
from src.verification.semantic_entropy import (
    generate_samples,
    cluster_by_entailment,
    compute_semantic_entropy,
    hallucination_risk_score,
)
from src.verification.ensemble_disagreement import compute_ensemble_disagreement
from src.verification.faithfulness import compute_faithfulness

logger = logging.getLogger(__name__)

# Keywords that indicate the query is about an uploaded document.
# When matched, the verifier enriches the LLM prompt with retrieved context.
_DOC_KEYWORDS = re.compile(
    r"\b(pdf|document|paper|uploaded|file|this text|the text|article|passage|excerpt)\b",
    re.IGNORECASE,
)

# Maximum number of source chunks to include in the enriched prompt
_MAX_CONTEXT_CHUNKS = 5
# Maximum characters per chunk in the enriched prompt
_MAX_CHUNK_CHARS = 600


def _classify_risk(score: float) -> str:
    """Classify a combined risk score into a human-readable label."""
    if score < 0.3:
        return "LOW"
    elif score < 0.6:
        return "MEDIUM"
    return "HIGH"


def _build_context_prompt(query: str, state: AgentState) -> tuple[str, bool]:
    """Build the prompt for semantic entropy / ensemble disagreement sampling.

    If the query contains document-reference keywords, the top retrieved
    chunks are prepended so the LLM can answer from context instead of
    complaining that it cannot see the file.

    Returns
    -------
    tuple[str, bool]
        (prompt_text, context_was_injected)
    """
    if not _DOC_KEYWORDS.search(query):
        return query, False

    # Collect source texts from local chunks and web results
    local_chunks = state.get("local_chunks", []) or []
    web_results = state.get("web_results", []) or []

    context_pieces: list[str] = []

    for chunk in local_chunks[:_MAX_CONTEXT_CHUNKS]:
        text = (chunk.get("text") or chunk.get("content") or "").strip()
        if text:
            context_pieces.append(text[:_MAX_CHUNK_CHARS])

    # Fill remaining slots with web results if local chunks are few
    remaining = _MAX_CONTEXT_CHUNKS - len(context_pieces)
    for result in web_results[:remaining]:
        text = (result.get("text") or result.get("content") or "").strip()
        if text:
            context_pieces.append(text[:_MAX_CHUNK_CHARS])

    if not context_pieces:
        return query, False

    context_block = "\n\n".join(context_pieces)
    enriched_prompt = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n---\n{context_block}\n---\n\n"
        f"Question: {query}"
    )
    logger.info(
        "Verifier: document-reference query detected — enriching prompt with "
        "%d context chunk(s) (%d chars total)",
        len(context_pieces),
        len(context_block),
    )
    return enriched_prompt, True


def verifier_node(state: AgentState) -> dict:
    """LangGraph node: verify the draft answer for hallucination risk.

    Runs semantic entropy, ensemble disagreement, and faithfulness checks,
    combines the scores, and returns the verification result.
    """
    settings = get_settings()
    query = state.get("refined_query") or state["original_query"]
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_retries)

    logger.info("Verifier checking answer (retry %d/%d) for: %s",
                retry_count, max_retries, query[:80])

    # ── Build context-aware prompt ────────────────────────────────────────
    sampling_prompt, context_injected = _build_context_prompt(query, state)

    # ── 1. Semantic Entropy ───────────────────────────────────────────────

    logger.info("Running semantic entropy analysis%s…",
                " (with context)" if context_injected else "")

    samples = generate_samples(
        prompt=sampling_prompt,
        n=settings.n_entropy_samples,
        temperature=settings.entropy_temperature,
    )
    clusters = cluster_by_entailment(samples)

    entropy = compute_semantic_entropy(clusters)
    n = len(samples)
    max_ent = math.log2(n) if n > 1 else 1.0
    entropy_risk = hallucination_risk_score(entropy, max_ent, n)

    logger.info("Entropy: %.4f bits (max %.4f), risk: %.2f%%",
                entropy, max_ent, entropy_risk * 100)

    # ── 2. Ensemble Disagreement ──────────────────────────────────────────

    logger.info("Running ensemble disagreement analysis%s…",
                " (with context)" if context_injected else "")
    disagreement_result = compute_ensemble_disagreement(sampling_prompt)

    disagreement_score = disagreement_result["disagreement_score"]

    logger.info("Ensemble disagreement: %.2f%%", disagreement_score * 100)

    # ── 3. Faithfulness ───────────────────────────────────────────────────

    logger.info("Running faithfulness check…")
    draft_answer = state.get("draft_answer", "")
    sources = state.get("sources", [])

    faithfulness_result = compute_faithfulness(
        draft_answer=draft_answer,
        sources=sources,
    )
    faithfulness_score = faithfulness_result["faithfulness_score"]

    logger.info("Faithfulness: %.2f%% unfaithful (%d/%d claims unsupported)",
                faithfulness_score * 100,
                faithfulness_result["unfaithful_claims"],
                faithfulness_result["total_claims"])

    # ── 4. Combined Score ─────────────────────────────────────────────────

    combined_risk = (
        0.5 * entropy_risk
        + 0.3 * disagreement_score
        + 0.2 * faithfulness_score
    )
    risk_label = _classify_risk(combined_risk)

    logger.info("Combined risk: %.2f%% (%s)", combined_risk * 100, risk_label)

    # ── 5. Build verification details ─────────────────────────────────────

    verification_details = {
        "context_injected": context_injected,
        "semantic_entropy": {
            "samples": samples,
            "clusters": [[s[:100] for s in c] for c in clusters],
            "n_clusters": len(clusters),
            "entropy": round(entropy, 4),
            "max_entropy": round(max_ent, 4),
            "risk_score": round(entropy_risk, 4),
        },
        "ensemble_disagreement": {
            "model1": settings.primary_model,
            "model2": settings.secondary_model,
            "model1_answer": disagreement_result.get("model1_answer", ""),
            "model2_answer": disagreement_result.get("model2_answer", ""),
            "model1_claims": disagreement_result.get("model1_claims", []),
            "model2_claims": disagreement_result.get("model2_claims", []),
            "contradictions_found": disagreement_result.get("contradictions_found", 0),
            "total_claim_pairs": disagreement_result.get("total_claim_pairs", 0),
            "disagreement_score": round(disagreement_score, 4),
        },
        "faithfulness": {
            "faithfulness_score": round(faithfulness_score, 4),
            "total_claims": faithfulness_result["total_claims"],
            "unfaithful_claims": faithfulness_result["unfaithful_claims"],
            "unfaithful_claim_texts": faithfulness_result["unfaithful_claim_texts"],
            "skipped": faithfulness_result["skipped"],
        },
        "weights": {
            "semantic_entropy": 0.5,
            "ensemble_disagreement": 0.3,
            "faithfulness": 0.2,
        },
    }

    return {
        "semantic_entropy_score": round(entropy_risk, 4),
        "ensemble_disagreement_score": round(disagreement_score, 4),
        "faithfulness_score": round(faithfulness_score, 4),
        "combined_risk_score": round(combined_risk, 4),
        "risk_label": risk_label,
        "verification_details": verification_details,
    }
