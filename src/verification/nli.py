"""
NLI Service — Local Transformers Model
=======================================

Runs Natural Language Inference using a locally loaded
``microsoft/deberta-large-mnli`` model via AutoTokenizer +
AutoModelForSequenceClassification + torch.softmax.

This is the correct way to run deberta-mnli inference — the
``pipeline("text-classification")`` wrapper does NOT handle
premise/hypothesis text-pairs correctly for this model class.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import get_settings

logger = logging.getLogger(__name__)

# ── Lazy-loaded model globals ─────────────────────────────────────────────────

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None


def _get_nli_model():
    """Lazy-load tokenizer + model once and reuse."""
    global _tokenizer, _model
    if _model is None:
        settings = get_settings()
        model_name = settings.hf_nli_model
        logger.info("Loading local NLI model: %s …", model_name)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.eval()
        logger.info("Local NLI model loaded successfully.")
    return _tokenizer, _model


# ── Core inference ────────────────────────────────────────────────────────────


def _run_nli_sync(premise: str, hypothesis: str) -> dict[str, float]:
    """Run NLI inference and return label → probability scores.

    Uses the direct AutoModel approach so the premise/hypothesis pair
    is tokenized correctly (as a sequence-pair input).
    """
    tokenizer, model = _get_nli_model()

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Read label names from model config (id2label is always set for mnli models)
    id2label: dict[int, str] = model.config.id2label
    scores = {
        id2label[i].upper(): probs[i].item()
        for i in range(len(probs))
    }
    return scores


# ── Public helpers ────────────────────────────────────────────────────────────


def get_nli_scores(
    premise: str,
    hypothesis: str,
) -> dict[str, float]:
    """Return raw NLI label scores (ENTAILMENT, NEUTRAL, CONTRADICTION)."""
    return _run_nli_sync(premise, hypothesis)


def entails(
    premise: str,
    hypothesis: str,
    threshold: Optional[float] = None,
) -> bool:
    """Return True if the NLI model considers premise → hypothesis as entailment."""
    if threshold is None:
        threshold = get_settings().nli_entailment_threshold

    scores = get_nli_scores(premise, hypothesis)
    result = scores.get("ENTAILMENT", 0.0) >= threshold
    logger.debug(
        "entails(%r, %r) -> %s  scores=%s",
        premise[:40], hypothesis[:40], result, scores,
    )
    return result


def contradicts(
    premise: str,
    hypothesis: str,
    threshold: Optional[float] = None,
) -> bool:
    """Return True if the NLI model considers premise → hypothesis as contradiction."""
    if threshold is None:
        threshold = get_settings().nli_contradiction_threshold

    scores = get_nli_scores(premise, hypothesis)
    result = scores.get("CONTRADICTION", 0.0) >= threshold
    logger.debug(
        "contradicts(%r, %r) -> %s  scores=%s",
        premise[:40], hypothesis[:40], result, scores,
    )
    return result
