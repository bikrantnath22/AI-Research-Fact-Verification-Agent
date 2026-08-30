"""
Embedding Service
=================

Loads the ``all-MiniLM-L6-v2`` sentence-transformer model and provides
synchronous helpers for embedding text.  The model is lazy-loaded on
first use and cached for the lifetime of the process.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import get_settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model (downloaded once, ~80 MB)."""
    global _model
    if _model is None:
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded (dim=%d)", settings.embedding_dimension)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single text string into a dense vector.

    Parameters
    ----------
    text : str
        The text to embed.

    Returns
    -------
    list[float]
        A dense vector of dimension ``embedding_dimension`` (384 for MiniLM).
    """
    model = _get_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into dense vectors.

    Parameters
    ----------
    texts : list[str]
        The texts to embed.

    Returns
    -------
    list[list[float]]
        A list of dense vectors, one per input text.
    """
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()
