"""
Verifier Agent
==============

Runs hallucination detection on the draft answer using two complementary
methods (ported from ``llm-hallu``):

1. **Semantic Entropy (60% weight):**
   Generate N samples at high temperature, cluster by bidirectional NLI
   entailment (HF Inference API), compute Shannon entropy → risk score.

2. **Ensemble Disagreement (40% weight):**
   Extract claims from answers by two different Groq models, check for
   NLI contradictions.

Combined: ``0.6 * entropy_risk + 0.4 * disagreement_score``
"""

from __future__ import annotations

import logging
import math

from src.config import get_settings
from src.graph.state import AgentState
from src.verification.semantic_entropy import (
    generate_samples,
    cluster_by_entailment,
    compute_semantic_entropy,
    hallucination_risk_score,
)
from src.verification.ensemble_disagreement import compute_ensemble_disagreement

logger = logging.getLogger(__name__)


def _classify_risk(score: float) -> str:
    """Classify a combined risk score into a human-readable label."""
    if score < 0.3:
        return "LOW"
    elif score < 0.6:
        return "MEDIUM"
    return "HIGH"


def verifier_node(state: AgentState) -> dict:
    """LangGraph node: verify the draft answer for hallucination risk.

    Runs both semantic entropy and ensemble disagreement, combines
    the scores, and returns the verification result.
    """
    settings = get_settings()
    query = state.get("refined_query") or state["original_query"]
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_retries)

    logger.info("Verifier checking answer (retry %d/%d) for: %s",
                retry_count, max_retries, query[:80])

    # ── 1. Semantic Entropy ──────────────────────────────────────────────

    logger.info("Running semantic entropy analysis…")

    samples = generate_samples(
        prompt=query,
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

    # ── 2. Ensemble Disagreement ─────────────────────────────────────────

    logger.info("Running ensemble disagreement analysis…")
    disagreement_result = compute_ensemble_disagreement(query)

    disagreement_score = disagreement_result["disagreement_score"]

    logger.info("Ensemble disagreement: %.2f%%", disagreement_score * 100)

    # ── 3. Combined Score ────────────────────────────────────────────────

    combined_risk = (0.6 * entropy_risk) + (0.4 * disagreement_score)
    risk_label = _classify_risk(combined_risk)

    logger.info("Combined risk: %.2f%% (%s)", combined_risk * 100, risk_label)

    # ── 4. Build verification details ────────────────────────────────────

    verification_details = {
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
        "weights": {"semantic_entropy": 0.6, "ensemble_disagreement": 0.4},
    }

    return {
        "semantic_entropy_score": round(entropy_risk, 4),
        "ensemble_disagreement_score": round(disagreement_score, 4),
        "combined_risk_score": round(combined_risk, 4),
        "risk_label": risk_label,
        "verification_details": verification_details,
    }
