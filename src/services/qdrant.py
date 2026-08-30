"""
Qdrant Service
==============

Wraps the ``qdrant-client`` SDK for collection management, chunk
upsertion, and similarity search.  One collection per document-upload
session (or a default collection for the system).
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.config import get_settings

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """Lazy-initialise the Qdrant client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        logger.info("Connected to Qdrant at %s:%d",
                     settings.qdrant_host, settings.qdrant_port)
    return _client


def create_collection(name: str) -> None:
    """Create a Qdrant collection with the configured vector dimension.

    Silently succeeds if the collection already exists.
    """
    settings = get_settings()
    client = _get_client()

    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        logger.info("Collection '%s' already exists — skipping creation", name)
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=settings.embedding_dimension,
            distance=Distance.COSINE,
        ),
    )
    logger.info("Created collection '%s' (dim=%d, cosine)",
                name, settings.embedding_dimension)


def upsert_chunks(
    collection: str,
    texts: list[str],
    vectors: list[list[float]],
    metadata: Optional[list[dict]] = None,
) -> int:
    """Upsert embedded chunks into a Qdrant collection.

    Parameters
    ----------
    collection : str
        Target collection name (must already exist).
    texts : list[str]
        The chunk texts.
    vectors : list[list[float]]
        Pre-computed embedding vectors, one per text.
    metadata : list[dict] | None
        Optional metadata dicts (e.g., ``{source: "file.pdf", page: 3}``).

    Returns
    -------
    int
        Number of points upserted.
    """
    client = _get_client()
    if metadata is None:
        metadata = [{}] * len(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": text, **meta},
        )
        for text, vec, meta in zip(texts, vectors, metadata)
    ]

    client.upsert(collection_name=collection, points=points)
    logger.info("Upserted %d chunks into '%s'", len(points), collection)
    return len(points)


def search(
    collection: str,
    query_vector: list[float],
    top_k: int = 5,
) -> list[dict]:
    """Search a collection for the nearest chunks.

    Parameters
    ----------
    collection : str
        Collection to search.
    query_vector : list[float]
        The embedded query vector.
    top_k : int
        Number of results to return.

    Returns
    -------
    list[dict]
        Each dict has keys: ``text``, ``score``, ``source``, plus any
        extra payload metadata.
    """
    client = _get_client()

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
    ).points

    chunks = []
    for point in results:
        payload = point.payload or {}
        chunks.append({
            "text": payload.get("text", ""),
            "score": point.score,
            "source": payload.get("source", "uploaded_document"),
            **{k: v for k, v in payload.items() if k not in ("text",)},
        })

    logger.info("Search returned %d results from '%s'", len(chunks), collection)
    return chunks


def delete_collection(name: str) -> None:
    """Delete a Qdrant collection."""
    client = _get_client()
    client.delete_collection(collection_name=name)
    logger.info("Deleted collection '%s'", name)


def collection_exists(name: str) -> bool:
    """Check whether a collection exists."""
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]
    return name in existing
