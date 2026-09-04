"""Secure ingestion primitives for user-provided knowledge documents."""

from .docx import (
    DOCX_MIME_TYPE,
    DocxExtractionError,
    DocxIngestionError,
    DocxIngestor,
    DocxValidationError,
)
from .models import DocumentSection, IngestedDocument, KnowledgeChunk

__all__ = [
    "DOCX_MIME_TYPE",
    "DocumentSection",
    "DocxExtractionError",
    "DocxIngestionError",
    "DocxIngestor",
    "DocxValidationError",
    "IngestedDocument",
    "KnowledgeChunk",
]
