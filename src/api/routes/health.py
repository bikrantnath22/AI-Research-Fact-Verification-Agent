"""
Health Check Route
==================

GET /health — verifies Qdrant connectivity and service status.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.api.models import HealthResponse
from src.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Quick health check — verifies Qdrant connectivity."""
    settings = get_settings()
    qdrant_ok = False

    try:
        from src.services.qdrant import _get_client
        client = _get_client()
        client.get_collections()
        qdrant_ok = True
    except Exception as e:
        logger.warning("Qdrant health check failed: %s", e)

    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        qdrant_connected=qdrant_ok,
        services={
            "qdrant": {"host": settings.qdrant_host, "port": settings.qdrant_port, "connected": qdrant_ok},
            "groq": {"model": settings.primary_model},
            "hf_nli": {"model": settings.hf_nli_model, "transport": "inference_api"},
        },
    )
