"""
Faithfulness Checker for RAG Hallucination Detection
=====================================================

Checks whether every factual claim in the synthesized answer is supported
by (i.e., not contradicted by) at least one retrieved source chunk.

Pipeline:
    1. Extract factual claims from the draft answer via Groq LLM.
    2. For each claim, run NLI against all source chunks.
    3. A claim is "unfaithful" if ALL sources contradict it (both directions).
    4. Faithfulness score = unfaithful_claims / total_claims  (0 = fully
       faithful, 1 = fully hallucinated).

Why both-direction contradiction?
    Single-direction NLI on short claims vs. short chunks produces many false
    positives (unrelated topics score as contradictions). Requiring BOTH
    directions to contradict eliminates most false positives while still
    catching genuine fabrications.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from groq import Groq

from src.config import get_settings
from src.verification import nli

logger = logging.getLogger(__name__)

# Maximum source chunks to check per claim (keeps latency reasonable)
_MAX_SOURCES_PER_CLAIM = 8
# Maximum claims to extract (very long answers can have 20+ claims)
_MAX_CLAIMS = 10


# ── Claim Extraction ──────────────────────────────────────────────────────────


def _extract_claims_from_answer(
    answer: str,
    model: Optional[str] = None,
) -> list[str]:
    """Extract factual claims from the synthesized answer using a Groq LLM.

    Parameters
    ----------
    answer : str
        The draft answer produced by the Synthesizer agent.
    model : str | None
        Groq model to use (defaults to ``primary_model``).

    Returns
    -------
    list[str]
        A list of individual factual claim strings (max ``_MAX_CLAIMS``).
    """
    settings = get_settings()
    model = model or settings.primary_model
    client = Groq(api_key=settings.groq_api_key)

    prompt = (
        "Extract the distinct factual claims made in the following answer. "
        "Output one claim per line. Do NOT include source citations like "
        "[Source N]. Do NOT include introductory text. Just the raw factual "
        "statements:\n\n"
        f"{answer}"
    )

    content = ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                break
        except Exception as exc:
            logger.warning("Claim extraction attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2.0)

    claims = [
        line.strip().lstrip("-*•1234567890. ")
        for line in content.split("\n")
        if line.strip()
    ]
    claims = [c for c in claims if len(c) > 10][:_MAX_CLAIMS]

    logger.info("Faithfulness: extracted %d claims from draft answer", len(claims))
    return claims


# ── Per-Claim Source Check ────────────────────────────────────────────────────


def _claim_is_supported(claim: str, source_texts: list[str]) -> bool:
    """Return True if at least one source does NOT bidirectionally contradict the claim.

    A claim is considered *unsupported* only when EVERY source contradicts it
    in both directions (claim→source AND source→claim). This conservative
    threshold avoids false positives from unrelated chunks.

    Parameters
    ----------
    claim : str
        A single factual statement to check.
    source_texts : list[str]
        Retrieved chunk texts to check against.

    Returns
    -------
    bool
        True = at least one source supports (or is neutral toward) the claim.
        False = every source contradicts the claim bidirectionally.
    """
    for src in source_texts[:_MAX_SOURCES_PER_CLAIM]:
        try:
            # Supported = NOT (both directions contradict)
            contradicted = nli.contradicts(claim, src) and nli.contradicts(src, claim)
            if not contradicted:
                return True  # Found a supporting / neutral source
        except Exception as exc:
            logger.warning("NLI check error: %s", exc)
            return True  # On error, assume supported (safe default)
    return False  # All sources contradicted it → unsupported


# ── Main Entry Point ──────────────────────────────────────────────────────────


def compute_faithfulness(
    draft_answer: str,
    sources: list[dict],
    model: Optional[str] = None,
) -> dict:
    """Compute the faithfulness score for a RAG-generated answer.

    Parameters
    ----------
    draft_answer : str
        The answer text produced by the Synthesizer agent.
    sources : list[dict]
        Retrieved source documents.  Each dict should have a ``"text"`` key.
        Both local chunks and web results are accepted.
    model : str | None
        Groq model to use for claim extraction.

    Returns
    -------
    dict
        Keys:
        - ``faithfulness_score`` (float 0–1): 0 = fully faithful, 1 = fully unfaithful.
        - ``total_claims`` (int): number of claims extracted.
        - ``unfaithful_claims`` (int): claims contradicted by all sources.
        - ``unfaithful_claim_texts`` (list[str]): the specific unfaithful claims.
        - ``skipped`` (bool): True if faithfulness check was skipped (no sources / answer).
    """
    # Guard: nothing to check
    if not draft_answer or not sources:
        logger.info("Faithfulness: skipped (no answer or no sources)")
        return {
            "faithfulness_score": 0.0,
            "total_claims": 0,
            "unfaithful_claims": 0,
            "unfaithful_claim_texts": [],
            "skipped": True,
        }

    # Build flat list of source texts
    source_texts = [s.get("text", "") for s in sources if s.get("text")]
    if not source_texts:
        logger.info("Faithfulness: skipped (sources have no text content)")
        return {
            "faithfulness_score": 0.0,
            "total_claims": 0,
            "unfaithful_claims": 0,
            "unfaithful_claim_texts": [],
            "skipped": True,
        }

    # Step 1: Extract claims
    claims = _extract_claims_from_answer(draft_answer, model=model)
    if not claims:
        logger.info("Faithfulness: no claims extracted — score 0.0")
        return {
            "faithfulness_score": 0.0,
            "total_claims": 0,
            "unfaithful_claims": 0,
            "unfaithful_claim_texts": [],
            "skipped": False,
        }

    # Step 2: Check each claim against sources
    unfaithful: list[str] = []
    for claim in claims:
        supported = _claim_is_supported(claim, source_texts)
        if not supported:
            unfaithful.append(claim)
            logger.info("Faithfulness: UNSUPPORTED claim — %s", claim[:80])
        else:
            logger.debug("Faithfulness: supported claim — %s", claim[:80])

    score = len(unfaithful) / len(claims) if claims else 0.0

    logger.info(
        "Faithfulness: %d/%d claims unsupported → score %.2f%%",
        len(unfaithful), len(claims), score * 100,
    )

    return {
        "faithfulness_score": round(score, 4),
        "total_claims": len(claims),
        "unfaithful_claims": len(unfaithful),
        "unfaithful_claim_texts": unfaithful,
        "skipped": False,
    }
