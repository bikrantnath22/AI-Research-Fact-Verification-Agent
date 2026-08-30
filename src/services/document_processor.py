"""
Document Processor
==================

Handles uploaded files (PDF and plain text), splits them into chunks,
and prepares them for embedding + Qdrant upsert.
"""

from __future__ import annotations

import logging
import uuid
from io import BytesIO
from typing import BinaryIO

from src.config import get_settings

logger = logging.getLogger(__name__)


def _chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into overlapping chunks by character count.

    Uses a simple sliding-window approach.  For a production system
    you'd want sentence-boundary-aware chunking, but this is sufficient
    for the portfolio demo.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf(file: BinaryIO) -> str:
    """Extract all text from a PDF file object."""
    from PyPDF2 import PdfReader

    reader = PdfReader(file)
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def process_document(
    file: BinaryIO,
    filename: str,
) -> tuple[str, list[dict]]:
    """Process an uploaded document into chunks ready for embedding.

    Parameters
    ----------
    file : BinaryIO
        The uploaded file object.
    filename : str
        Original filename (used to detect format and as metadata).

    Returns
    -------
    tuple[str, list[dict]]
        - A collection ID (``str``) unique to this upload.
        - A list of chunk dicts: ``{text: str, metadata: {source, chunk_index}}``.
    """
    settings = get_settings()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Extract raw text
    if ext == "pdf":
        logger.info("Extracting text from PDF: %s", filename)
        raw_text = extract_text_from_pdf(file)
    else:
        logger.info("Reading plain text: %s", filename)
        raw_text = file.read().decode("utf-8", errors="replace")

    if not raw_text.strip():
        logger.warning("Empty document: %s", filename)
        return "", []

    # Chunk
    chunks = _chunk_text(raw_text, settings.chunk_size, settings.chunk_overlap)
    logger.info("Chunked '%s' into %d pieces (size=%d, overlap=%d)",
                filename, len(chunks), settings.chunk_size, settings.chunk_overlap)

    # Build structured output
    collection_id = f"doc_{uuid.uuid4().hex[:12]}"
    chunk_dicts = [
        {
            "text": chunk,
            "metadata": {
                "source": filename,
                "chunk_index": i,
                "collection": collection_id,
            },
        }
        for i, chunk in enumerate(chunks)
    ]

    return collection_id, chunk_dicts
