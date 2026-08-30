"""
Semantic Entropy for Hallucination Detection
=============================================

Adapted from ``llm-hallu/semantic_entropy.py``.

Pipeline:
    1. Generate N samples for the same prompt via Groq.
    2. Cluster samples by bidirectional entailment (local NLI model via transformers).
    3. Compute Shannon entropy over the cluster-size distribution.
    4. Normalize entropy to a 0–1 hallucination risk score.
"""

from __future__ import annotations

import math
import logging
import time
import re
from collections import defaultdict
from itertools import combinations
from typing import Optional

from groq import Groq

from src.config import get_settings

logger = logging.getLogger(__name__)


# ── 1. Generation ────────────────────────────────────────────────────────────


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from Qwen/DeepSeek models."""
    cleaned = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
    return cleaned.strip()

def generate_samples(
    prompt: str,
    model: Optional[str] = None,
    n: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 1024,
) -> list[str]:
    """Generate *n* sampled completions for a single prompt via the Groq API.

    Parameters
    ----------
    prompt : str
        The user-facing question / instruction.
    model : str | None
        Groq-hosted model identifier (defaults to ``PRIMARY_MODEL``).
    n : int | None
        Number of independent samples (defaults to ``N_ENTROPY_SAMPLES``).
    temperature : float | None
        Sampling temperature (defaults to ``ENTROPY_TEMPERATURE``).
    max_tokens : int
        Maximum tokens per completion.

    Returns
    -------
    list[str]
        A list of *n* completion strings.
    """
    settings = get_settings()
    model = model or settings.primary_model
    n = n or settings.n_entropy_samples
    temperature = temperature if temperature is not None else settings.entropy_temperature

    client = Groq(api_key=settings.groq_api_key)
    samples: list[str] = []

    for i in range(n):
        text = ""
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer the user's question concisely in 1-3 sentences.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=2048,
                )
                raw_text = (response.choices[0].message.content or "").strip()
                text = strip_think_tags(raw_text)
                if len(text) > 5:
                    break
            except Exception as e:
                logger.warning("Generation attempt %d failed: %s", attempt + 1, e)
            time.sleep(5.0)
            
        if not text or len(text) <= 5:
            logger.warning("Failed to generate a valid sample for index %d", i)
            continue
            
        samples.append(text)
        logger.warning("DEBUG sample %d: '%s'", i, text[:50].replace('\n', ' '))

        # Rate-limit pause between calls
        if i < n - 1:
            time.sleep(5.0)

    logger.warning("Generated %d samples for entropy analysis", len(samples))
    return samples


# ── 2. Hybrid Clustering (Embedding Similarity + NLI Contradiction Veto) ─────


def cluster_by_entailment(
    samples: list[str],
    threshold: Optional[float] = None,
) -> list[list[str]]:
    """Cluster samples using a hybrid of embedding similarity + NLI contradiction veto.

    Two-stage process:
    1. **Embedding similarity** (``all-MiniLM-L6-v2``): two samples are
       candidate-compatible if their cosine similarity ≥ ``sim_threshold``
       (default 0.80). This groups same-topic paraphrases together.
    2. **NLI contradiction veto** (``deberta-large-mnli``): if either direction
       of the candidate pair is a contradiction, they are placed in separate
       clusters — even if they share topic vocabulary.

    Why hybrid?
    - Pure embedding similarity: groups "no life detected" + "detected life on
      TRAPPIST" in the same cluster (both are JWST sentences → high cosine
      similarity) → **misses hallucinations**.
    - Pure NLI entailment: splits "Kapton film" + "polyimide Kapton film" into
      different clusters (strict entailment fails on paraphrases) →
      **false HIGH entropy on factual answers**.
    - Hybrid: embeddings catch paraphrases; NLI contradiction veto catches
      factual disagreements.

    Parameters
    ----------
    samples : list[str]
        The generated text completions to cluster.
    threshold : float | None
        Cosine similarity threshold (default 0.80). Pairs below this are
        placed in separate clusters without running NLI.

    Returns
    -------
    list[list[str]]
        Each inner list is a cluster of semantically equivalent samples.
    """
    import numpy as np
    from src.services.embeddings import embed_batch
    from src.verification import nli

    n = len(samples)
    if n == 0:
        return []
    if n == 1:
        return [samples]

    sim_threshold = threshold if threshold is not None else 0.80

    logger.info(
        "Hybrid clustering %d samples (sim_threshold=%.2f)…", n, sim_threshold
    )

    # ── Stage 1: Embedding similarity ────────────────────────────────────────
    vectors = np.array(embed_batch(samples))          # (n, dim)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = vectors / norms
    sim_matrix = normed @ normed.T                    # (n, n)

    adj: dict[int, set[int]] = defaultdict(set)
    pairs = list(combinations(range(n), 2))

    for i, j in pairs:
        sim = float(sim_matrix[i, j])
        if sim < sim_threshold:
            logger.debug("Pair (%d,%d) sim=%.3f < threshold → skip", i, j, sim)
            continue

        # ── Stage 2: NLI contradiction veto ──────────────────────────────────
        # If either direction is a contradiction, keep them in separate clusters
        contradicted = (
            nli.contradicts(samples[i], samples[j])
            or nli.contradicts(samples[j], samples[i])
        )
        if contradicted:
            logger.info(
                "Pair (%d,%d) sim=%.3f but NLI contradiction detected → separate clusters",
                i, j, sim,
            )
            continue

        logger.debug("Pair (%d,%d) sim=%.3f, no contradiction → same cluster", i, j, sim)
        adj[i].add(j)
        adj[j].add(i)

    # ── BFS connected components ──────────────────────────────────────────────
    visited: set[int] = set()
    clusters: list[list[str]] = []

    for start in range(n):
        if start in visited:
            continue
        component: list[int] = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            queue.extend(adj[node] - visited)
        clusters.append([samples[idx] for idx in component])

    logger.info("Hybrid clustering: %d clusters from %d samples", len(clusters), n)
    return clusters



# ── 3. Entropy ───────────────────────────────────────────────────────────────


def compute_semantic_entropy(clusters: list[list[str]]) -> float:
    """Compute Shannon entropy over the cluster-size distribution.

    H = -Σ (p_k · log₂ p_k)

    where p_k = |cluster_k| / total_samples.

    Returns
    -------
    float
        Shannon entropy in bits.  0 = all samples in one cluster (perfect
        agreement), log₂(N) = every sample in its own cluster (maximum
        disagreement).
    """
    total = sum(len(c) for c in clusters)
    if total == 0:
        return 0.0

    entropy = 0.0
    for cluster in clusters:
        p = len(cluster) / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ── 4. Risk Score ────────────────────────────────────────────────────────────


def hallucination_risk_score(
    entropy: float,
    max_entropy: Optional[float] = None,
    n_samples: int = 10,
) -> float:
    """Normalize entropy to a 0–1 hallucination risk score.

    Parameters
    ----------
    entropy : float
        The Shannon entropy returned by :func:`compute_semantic_entropy`.
    max_entropy : float | None
        Upper bound for normalization.  If *None*, defaults to log₂(n_samples).
    n_samples : int
        Number of samples (used to compute default *max_entropy*).

    Returns
    -------
    float
        A value in [0, 1] where 0 = no risk and 1 = maximum risk.
    """
    if max_entropy is None:
        max_entropy = math.log2(n_samples) if n_samples > 1 else 1.0
    if max_entropy == 0:
        return 0.0
    return min(entropy / max_entropy, 1.0)
