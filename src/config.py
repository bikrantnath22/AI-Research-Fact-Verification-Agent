"""
Centralized Configuration
=========================

All settings loaded from environment variables / .env file via Pydantic BaseSettings.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    """Application-wide configuration. Values come from .env or env vars."""

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────────────────
    groq_api_key: str = ""
    tavily_api_key: str = ""
    hf_api_token: str = ""

    # ── LLM Models (Groq-hosted) ─────────────────────────────────────────
    primary_model: str = "openai/gpt-oss-20b"
    secondary_model: str = "openai/gpt-oss-120b"

    # ── Qdrant ────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ── NLI (Local Transformers Model) ───────────────────────────────────
    hf_nli_model: str = "microsoft/deberta-large-mnli"
    nli_entailment_threshold: float = 0.5
    nli_contradiction_threshold: float = 0.5
    nli_max_concurrent: int = 5

    # ── Verification Thresholds ──────────────────────────────────────────
    risk_threshold: float = 0.6
    max_retries: int = 2
    retrieval_confidence_threshold: float = 0.5
    n_entropy_samples: int = 5
    entropy_temperature: float = 0.9

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Document Processing ──────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── MCP ───────────────────────────────────────────────────────────────
    mcp_qdrant_url: str = "http://mcp-qdrant:8001/mcp"
    mcp_tavily_url: str = "http://mcp-tavily:8002/mcp"
    mcp_transport: str = "http"  # "http" for Docker/k3s, "stdio" for local dev


def get_settings() -> Settings:
    """Factory that creates a cached Settings instance."""
    return Settings()
