"""
Pydantic Request / Response Models
===================================

Schemas for the FastAPI endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The research question to answer.",
        examples=["What is retrieval-augmented generation?"],
    )
    collection: str = Field(
        default="",
        description="Optional Qdrant collection name (from a prior document upload).",
    )


# ── Response Models ───────────────────────────────────────────────────────────


class SourceInfo(BaseModel):
    """A single source used in the answer."""

    id: int
    type: str  # "local" | "web"
    text: str
    ref: str
    title: str = ""
    url: str = ""
    score: float = 0.0


class VerificationDetails(BaseModel):
    """Full verification breakdown."""

    semantic_entropy: dict = {}
    ensemble_disagreement: dict = {}
    weights: dict = {"semantic_entropy": 0.6, "ensemble_disagreement": 0.4}


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    question: str
    answer: str
    sources: list[SourceInfo] = []
    used_web_search: bool = False
    retrieval_confidence: float = 0.0

    # Verification
    combined_risk_score: float = 0.0
    risk_label: str = "LOW"
    semantic_entropy_score: float = 0.0
    ensemble_disagreement_score: float = 0.0
    verification_details: VerificationDetails = VerificationDetails()

    # Loop metadata
    retry_count: int = 0
    refined_query: str = ""


class UploadResponse(BaseModel):
    """Response body for POST /upload."""

    collection_id: str
    filename: str
    n_chunks: int
    status: str = "success"


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    qdrant_connected: bool = False
    services: dict = {}
