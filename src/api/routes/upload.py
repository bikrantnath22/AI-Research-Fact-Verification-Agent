"""
Document Upload Route
======================

POST /upload — accepts a file upload (PDF or TXT), chunks and embeds
the document, and upserts into a new Qdrant collection.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, UploadFile, HTTPException

from src.api.models import UploadResponse
from src.services.document_processor import process_document
from src.services.embeddings import embed_batch
from src.services.qdrant import create_collection, upsert_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF or TXT), chunk it, embed, and store in Qdrant.

    Returns the collection ID that can be passed to ``POST /query`` to
    restrict retrieval to this document's content.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("pdf", "txt", "md", "text"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Accepted: .pdf, .txt, .md",
        )

    logger.info("Upload received: %s (%s)", file.filename, file.content_type)

    try:
        collection_id, chunk_dicts = process_document(file.file, file.filename)

        if not chunk_dicts:
            raise HTTPException(status_code=400, detail="Document is empty or could not be parsed")

        # Embed all chunks
        texts = [c["text"] for c in chunk_dicts]
        vectors = embed_batch(texts)

        # Create collection and upsert
        create_collection(collection_id)
        metadata = [c["metadata"] for c in chunk_dicts]
        upsert_chunks(
            collection=collection_id,
            texts=texts,
            vectors=vectors,
            metadata=metadata,
        )

        logger.info("Uploaded '%s' → collection '%s' (%d chunks)",
                     file.filename, collection_id, len(chunk_dicts))

        return UploadResponse(
            collection_id=collection_id,
            filename=file.filename,
            n_chunks=len(chunk_dicts),
            status="success",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {e}")
