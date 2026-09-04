from __future__ import annotations

import io
import re
import zipfile
from hashlib import sha256
from pathlib import PurePosixPath

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .models import DocumentSection, IngestedDocument, KnowledgeChunk


DOCX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
EXTRACTOR_VERSION = "docx-1"


class DocxIngestionError(ValueError):
    pass


class DocxValidationError(DocxIngestionError):
    pass


class DocxExtractionError(DocxIngestionError):
    pass


def _clean(value: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []
    output: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            minimum = start + max(chunk_size // 2, overlap + 1)
            boundary = max(
                text.rfind("\n", minimum, end),
                text.rfind(". ", minimum, end),
                text.rfind(" ", minimum, end),
            )
            if boundary >= minimum:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            output.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return output


class DocxIngestor:
    def __init__(
        self,
        *,
        max_bytes: int = 15 * 1024 * 1024,
        max_uncompressed_bytes: int = 60 * 1024 * 1024,
        chunk_size: int = 1_600,
        chunk_overlap: int = 200,
    ) -> None:
        self.max_bytes = int(max_bytes)
        self.max_uncompressed_bytes = int(max_uncompressed_bytes)
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        if self.max_bytes < 1 or self.max_uncompressed_bytes < self.max_bytes:
            raise ValueError("invalid DOCX size limits")
        if self.chunk_size < 200 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("invalid DOCX chunk configuration")

    def _validate_archive(self, data: bytes, filename: str, content_type: str) -> None:
        if not filename.casefold().endswith(".docx"):
            raise DocxValidationError("Only .docx files are accepted")
        if content_type and content_type.casefold() not in {
            DOCX_MIME_TYPE,
            "application/octet-stream",
        }:
            raise DocxValidationError("The uploaded file is not a DOCX document")
        if not data or len(data) > self.max_bytes:
            raise DocxValidationError(
                f"DOCX size must be between 1 and {self.max_bytes} bytes"
            )
        if not zipfile.is_zipfile(io.BytesIO(data)):
            raise DocxValidationError("Invalid DOCX archive")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                infos = archive.infolist()
                names = {info.filename for info in infos}
                if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                    raise DocxValidationError("Invalid Word document structure")
                if any(name.casefold().endswith("vbaproject.bin") for name in names):
                    raise DocxValidationError("Macro-enabled documents are not accepted")
                if len(infos) > 5_000:
                    raise DocxValidationError("DOCX archive contains too many entries")
                total = 0
                for info in infos:
                    path = PurePosixPath(info.filename.replace("\\", "/"))
                    if path.is_absolute() or ".." in path.parts:
                        raise DocxValidationError("Unsafe path in DOCX archive")
                    if info.flag_bits & 0x1:
                        raise DocxValidationError("Encrypted DOCX files are not accepted")
                    total += info.file_size
                    if total > self.max_uncompressed_bytes:
                        raise DocxValidationError("DOCX archive expands beyond the safe limit")
        except zipfile.BadZipFile as exc:
            raise DocxValidationError("Invalid DOCX archive") from exc

    @staticmethod
    def _heading_level(paragraph: Paragraph) -> int | None:
        style_name = str(getattr(paragraph.style, "name", "") or "")
        match = re.search(r"(?:heading|titre)\s*(\d+)$", style_name, re.I)
        if match:
            return max(1, min(9, int(match.group(1))))
        return None

    def ingest(
        self,
        data: bytes,
        *,
        filename: str,
        content_type: str,
        module_namespace: str,
        title: str | None = None,
    ) -> IngestedDocument:
        namespace = module_namespace.strip().casefold()
        if not namespace:
            raise DocxValidationError("A module is required")
        safe_filename = PurePosixPath(filename.replace("\\", "/")).name
        self._validate_archive(data, safe_filename, content_type)
        try:
            document = Document(io.BytesIO(data))
        except Exception as exc:
            raise DocxExtractionError("The DOCX content cannot be read") from exc

        document_title = _clean(title or document.core_properties.title or "")
        if not document_title:
            document_title = PurePosixPath(safe_filename).stem.strip() or "Document"

        sections: list[DocumentSection] = []
        stack: list[tuple[int, str]] = []
        current_heading = document_title
        current_level = 1
        current_path = (document_title,)
        body: list[str] = []
        paragraph_count = 0
        table_count = 0

        def flush() -> None:
            text = _clean("\n".join(body))
            if text:
                sections.append(
                    DocumentSection(
                        heading=current_heading,
                        heading_path=current_path,
                        level=current_level,
                        position=len(sections),
                        text=text,
                    )
                )

        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                paragraph = Paragraph(child, document)
                text = _clean(paragraph.text)
                if not text:
                    continue
                paragraph_count += 1
                level = self._heading_level(paragraph)
                if level is not None:
                    flush()
                    body = []
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, text))
                    current_heading = text
                    current_level = level
                    current_path = tuple(item[1] for item in stack)
                else:
                    body.append(text)
            elif child.tag.endswith("}tbl"):
                table = Table(child, document)
                rows: list[str] = []
                for row in table.rows:
                    cells = [_clean(cell.text) for cell in row.cells]
                    line = " | ".join(cell for cell in cells if cell)
                    if line:
                        rows.append(line)
                if rows:
                    body.append("\n".join(rows))
                    table_count += 1
        flush()

        extracted_text = "\n\n".join(section.text for section in sections).strip()
        if not extracted_text:
            raise DocxExtractionError("The DOCX contains no indexable text")

        chunks: list[KnowledgeChunk] = []
        for section in sections:
            for piece in _windows(
                section.text, self.chunk_size, self.chunk_overlap
            ):
                position = len(chunks)
                section_path = " > ".join(section.heading_path)
                content_hash = sha256(piece.encode("utf-8")).hexdigest()
                chunk_id = sha256(
                    "\x1f".join(
                        (namespace, document_title, section_path, str(position), content_hash)
                    ).encode("utf-8")
                ).hexdigest()
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        section=section.heading,
                        section_path=section.heading_path,
                        position=position,
                        text=piece,
                        embedding_text=(
                            f"Module: {namespace}\nDocument: {document_title}\n"
                            f"Section: {section_path}\n{piece}"
                        ),
                        content_hash=content_hash,
                    )
                )

        return IngestedDocument(
            file_sha256=sha256(data).hexdigest(),
            text_sha256=sha256(extracted_text.encode("utf-8")).hexdigest(),
            title=document_title,
            module_namespace=namespace,
            original_filename=safe_filename,
            mime_type=DOCX_MIME_TYPE,
            size_bytes=len(data),
            extracted_text=extracted_text,
            extracted_characters=len(extracted_text),
            sections=tuple(sections),
            chunks=tuple(chunks),
            paragraph_count=paragraph_count,
            table_count=table_count,
            extractor_version=EXTRACTOR_VERSION,
        )
