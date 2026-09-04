from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """A normalized, ordered section extracted from an uploaded document."""

    heading: str
    heading_path: tuple[str, ...]
    level: int
    position: int
    text: str


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """A deterministic chunk ready to be embedded and persisted."""

    chunk_id: str
    section: str
    section_path: tuple[str, ...]
    position: int
    text: str
    embedding_text: str
    content_hash: str

    @property
    def id(self) -> str:
        """Compatibility alias for integrations that expose an ``id`` field."""

        return self.chunk_id


@dataclass(frozen=True, slots=True)
class IngestedDocument:
    """Validated DOCX content and its deterministic embedding inputs."""

    file_sha256: str
    text_sha256: str
    title: str
    module_namespace: str
    original_filename: str
    mime_type: str
    size_bytes: int
    extracted_text: str
    extracted_characters: int
    sections: tuple[DocumentSection, ...]
    chunks: tuple[KnowledgeChunk, ...]
    paragraph_count: int
    table_count: int
    extractor_version: str
