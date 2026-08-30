"""
FastAPI Application
====================

Application factory with CORS, lifespan, and route registration.
Serves the frontend at ``/`` and API endpoints under ``/api``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from src.api.routes import health, upload, query

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle.

    - Warm up the HuggingFace Inference API with a test NLI call.
    - Pre-load the embedding model.
    """
    logger.info("Starting Research Agent API…")

    # Warm up embedding model
    try:
        from src.services.embeddings import embed_text
        embed_text("warmup")
        logger.info("Embedding model warmed up")
    except Exception as e:
        logger.warning("Embedding warmup failed (non-fatal): %s", e)

    # Warm up Local NLI Model
    try:
        from src.verification.nli import get_nli_scores
        scores = await get_nli_scores("The sky is blue.", "The sky has a blue color.")
        logger.info("Local NLI model warmed up: %s", scores)
    except Exception as e:
        logger.warning("Local NLI warmup failed (non-fatal): %s", e)

    yield

    logger.info("Shutting down Research Agent API")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="AI Research & Fact-Verification Agent",
        description=(
            "Multi-agent system that plans queries, retrieves from Qdrant + web, "
            "synthesizes answers with source attribution, and verifies for "
            "hallucination risk using semantic entropy + ensemble disagreement."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ──
    app.include_router(health.router, tags=["Health"])
    app.include_router(upload.router, tags=["Documents"])
    app.include_router(query.router, tags=["Query"])

    # ── Frontend ──
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Serve the single-page frontend dashboard."""
        index = _FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(index, media_type="text/html")
        return {"message": "Frontend not found. Place index.html in frontend/"}

    return app


# ── Module-level app instance for uvicorn ────────────────────────────────────

app = create_app()

# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
