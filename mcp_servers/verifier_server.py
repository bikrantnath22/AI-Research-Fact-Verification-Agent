"""
MCP Server — Verification Pipeline (Stretch Goal)
===================================================

Exposes the full hallucination verification pipeline as an MCP tool.
External MCP-compatible clients (Claude Desktop, Claude Code, etc.) can
invoke ``verify_answer`` to get a risk score for any question/answer pair.

Tools:
- ``verify_answer``: Run semantic entropy + ensemble disagreement on a
  question/answer pair and return the risk assessment.
"""



import asyncio
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP

from src.config import get_settings
from src.verification.semantic_entropy import (
    generate_samples,
    cluster_by_entailment,
    compute_semantic_entropy,
    hallucination_risk_score,
)
from src.verification.ensemble_disagreement import compute_ensemble_disagreement

# ── MCP Server Setup ─────────────────────────────────────────────────────────

mcp = FastMCP("Verification Pipeline Server")


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def verify_answer(
    question: str,
    answer: str,
    n_samples: int = 5,
) -> str:
    """Verify an LLM-generated answer for hallucination risk.

    Runs the full verification pipeline:
    1. Semantic entropy: generate N samples, cluster by NLI entailment,
       compute entropy → risk score (60% weight).
    2. Ensemble disagreement: compare claims from two models for NLI
       contradictions (40% weight).

    Args:
        question: The original question that produced the answer.
        answer: The LLM-generated answer to verify.
        n_samples: Number of samples for entropy analysis (default: 5).

    Returns:
        JSON string with combined_risk_score, risk_label, and full details.
    """
    settings = get_settings()

    # Semantic entropy
    samples = generate_samples(
        prompt=question,
        n=n_samples,
        temperature=settings.entropy_temperature,
    )

    clusters = cluster_by_entailment(samples)
    entropy = compute_semantic_entropy(clusters)
    max_ent = math.log2(n_samples) if n_samples > 1 else 1.0
    entropy_risk = hallucination_risk_score(entropy, max_ent, n_samples)

    # Ensemble disagreement
    disagreement_result = compute_ensemble_disagreement(question)
    disagreement_score = disagreement_result["disagreement_score"]

    # Combined
    combined_risk = (0.6 * entropy_risk) + (0.4 * disagreement_score)
    risk_label = "LOW" if combined_risk < 0.3 else "MEDIUM" if combined_risk < 0.6 else "HIGH"

    result = {
        "question": question,
        "answer": answer,
        "combined_risk_score": round(combined_risk, 4),
        "risk_label": risk_label,
        "semantic_entropy_risk": round(entropy_risk, 4),
        "ensemble_disagreement_score": round(disagreement_score, 4),
        "n_clusters": len(clusters),
        "entropy_bits": round(entropy, 4),
        "contradictions_found": disagreement_result.get("contradictions_found", 0),
    }

    return json.dumps(result, default=str)


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verification Pipeline MCP Server")
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default=os.getenv("MCP_TRANSPORT", "http"),
        help="MCP transport: 'http' (Streamable HTTP) or 'stdio'",
    )
    parser.add_argument("--port", type=int, default=8003, help="HTTP port (default: 8003)")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = args.port
        mcp.run(transport="sse")
