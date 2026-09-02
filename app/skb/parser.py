from __future__ import annotations

import html
import re
from dataclasses import dataclass

from app.skb.models import (
    DocumentChunk,
    WikiPage,
    WikiSection,
    content_digest,
)


_HEADING_LINE = re.compile(r"^\s*(={2,})\s*(.*?)\s*=+\s*$")
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_CODE_BLOCK = re.compile(r"<(?:code|file)(?:\s+[^>]*)?>(.*?)</(?:code|file)>", re.I | re.S)
_NOWIKI_BLOCK = re.compile(r"<(?:nowiki)>(.*?)</(?:nowiki)>", re.I | re.S)
_MEDIA = re.compile(r"\{\{\s*([^{}|?]+)(?:\?[^|}]*)?(?:\|([^{}]*))?\s*\}\}")
_LINK = re.compile(r"\[\[\s*([^\]|]+)(?:\|([^\]]+))?\s*\]\]")
_FOOTNOTE = re.compile(r"\(\((.*?)\)\)", re.S)


@dataclass(slots=True)
class _ChunkDraft:
    section_index: int
    section_position: int
    section: WikiSection
    text: str


def _protect_blocks(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        key = f"\x00SKB{len(protected)}\x00"
        protected[key] = match.group(1).strip()
        return key

    text = _CODE_BLOCK.sub(replace, text)
    text = _NOWIKI_BLOCK.sub(replace, text)
    return text, protected


def clean_dokuwiki_markup(text: str) -> str:
    """Convert common DokuWiki markup into embedding-friendly plain text."""

    text = _COMMENT.sub("", text)
    text, protected = _protect_blocks(text)

    def media_text(match: re.Match[str]) -> str:
        label = (match.group(2) or "").strip()
        if label:
            return label
        target = match.group(1).strip().rsplit(":", 1)[-1]
        return re.sub(r"[_-]+", " ", target.rsplit(".", 1)[0])

    def link_text(match: re.Match[str]) -> str:
        return (match.group(2) or match.group(1)).strip()

    text = _MEDIA.sub(media_text, text)
    text = _LINK.sub(link_text, text)
    text = _FOOTNOTE.sub(lambda match: f" ({match.group(1).strip()}) ", text)
    text = re.sub(r"~~[^~]+~~", " ", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</?(?:del|sub|sup|kbd|mark|abbr|acronym)[^>]*>", "", text, flags=re.I)
    text = re.sub(r"(?<!:)//(.*?)(?<!:)//", r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("''", "")
    text = re.sub(r"\\\\(?=\s|$)", "\n", text)

    output_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"-{4,}", stripped):
            continue
        stripped = re.sub(r"^[ \t]*[*-][ \t]+", "- ", stripped)
        if stripped.startswith(("|", "^")) and stripped.endswith(("|", "^")):
            cells = [cell.strip() for cell in re.split(r"[|^]", stripped) if cell.strip()]
            stripped = " | ".join(cells)
        stripped = re.sub(r"[ \t]+", " ", stripped).strip()
        output_lines.append(stripped)

    cleaned = "\n".join(output_lines)
    for key, value in protected.items():
        cleaned = cleaned.replace(key, value)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def parse_sections(page: WikiPage) -> list[WikiSection]:
    """Parse raw DokuWiki markup into hierarchical, plain-text sections."""

    sections: list[WikiSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading = page.title
    current_level = 1
    current_path: tuple[str, ...] = (page.title,)
    body: list[str] = []

    def flush() -> None:
        cleaned = clean_dokuwiki_markup("\n".join(body))
        if cleaned:
            sections.append(
                WikiSection(
                    heading=current_heading,
                    heading_path=current_path,
                    level=current_level,
                    text=cleaned,
                )
            )

    for line in page.raw_text.splitlines():
        match = _HEADING_LINE.match(line)
        if not match:
            body.append(line)
            continue

        flush()
        body = []
        marks, raw_heading = match.groups()
        heading = clean_dokuwiki_markup(raw_heading) or page.title
        level = 7 - min(6, len(marks))
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading))
        current_heading = heading
        current_level = level
        current_path = tuple(item[1] for item in heading_stack)

    flush()
    return sections


def _windows(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        hard_end = min(length, start + chunk_size)
        end = hard_end
        if hard_end < length:
            minimum_break = start + max(chunk_size // 2, overlap + 1)
            candidates = [
                text.rfind("\n\n", minimum_break, hard_end),
                text.rfind("\n", minimum_break, hard_end),
                text.rfind(". ", minimum_break, hard_end),
                text.rfind(" ", minimum_break, hard_end),
            ]
            boundary = max(candidates)
            if boundary >= minimum_break:
                end = boundary + (1 if text[boundary] != "." else 1)

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        # Avoid starting halfway through a word while keeping bounded overlap.
        if next_start and not text[next_start - 1].isspace():
            whitespace = text.find(" ", next_start, min(end, next_start + 80))
            if whitespace >= 0:
                next_start = whitespace + 1
        start = next_start
    return chunks


def _merge_short_drafts(
    drafts: list[_ChunkDraft], *, chunk_size: int, min_chunk_size: int
) -> list[_ChunkDraft]:
    """Merge useful short sections into a neighbour and discard isolated noise.

    DokuWiki contains many media-only pages whose cleaned body is just a short
    filename.  Indexing those fragments produces high-similarity false positives.
    A short fragment is therefore retained only when it can be attached to an
    adjacent, meaningful fragment without making an unbounded chunk.
    """

    output = list(drafts)
    max_merged_size = chunk_size + min_chunk_size
    index = 0
    while index < len(output):
        current = output[index]
        if len(current.text) >= min_chunk_size:
            index += 1
            continue

        if index > 0:
            previous = output[index - 1]
            label = (
                f"{current.section.heading}:\n"
                if current.section.heading_path != previous.section.heading_path
                else ""
            )
            merged = f"{previous.text}\n\n{label}{current.text}".strip()
            if len(merged) <= max_merged_size:
                previous.text = merged
                output.pop(index)
                continue

        if index + 1 < len(output):
            following = output[index + 1]
            label = (
                f"{current.section.heading}:\n"
                if current.section.heading_path != following.section.heading_path
                else ""
            )
            merged = f"{label}{current.text}\n\n{following.text}".strip()
            if len(merged) <= max_merged_size:
                following.text = merged
                output.pop(index)
                continue

        # A standalone fragment below the configured minimum is usually a media
        # filename, navigation remnant, or otherwise too weak to ground an answer.
        output.pop(index)

    return output


def chunk_page(
    page: WikiPage,
    *,
    chunk_size: int = 1_600,
    chunk_overlap: int = 200,
    min_chunk_size: int = 80,
) -> list[DocumentChunk]:
    """Create deterministic, section-aware chunks from a DokuWiki page."""

    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    if min_chunk_size < 1 or min_chunk_size > chunk_size:
        raise ValueError("min_chunk_size must be between 1 and chunk_size")

    drafts: list[_ChunkDraft] = []
    for section_index, section in enumerate(parse_sections(page)):
        pieces = _windows(section.text, chunk_size=chunk_size, overlap=chunk_overlap)
        if len(pieces) > 1 and len(pieces[-1]) < min_chunk_size:
            tail = pieces.pop()
            pieces[-1] = f"{pieces[-1]}\n{tail}".strip()

        drafts.extend(
            _ChunkDraft(
                section_index=section_index,
                section_position=section_position,
                section=section,
                text=piece,
            )
            for section_position, piece in enumerate(pieces)
        )

    drafts = _merge_short_drafts(
        drafts,
        chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
    )

    chunks: list[DocumentChunk] = []
    for position, draft in enumerate(drafts):
        section = draft.section
        piece = draft.text
        chunk_hash = content_digest(piece)
        section_path_text = " > ".join(section.heading_path)
        chunk_id = content_digest(
            page.page_id,
            str(draft.section_index),
            str(draft.section_position),
            section_path_text,
            chunk_hash,
        )
        context = [f"Page: {page.title}"]
        if page.module:
            context.insert(0, f"Module: {page.module}")
        if section_path_text:
            context.append(f"Section: {section_path_text}")
        context.append(piece)
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                page_id=page.page_id,
                title=page.title,
                source_url=page.source_url,
                module=page.module,
                section=section.heading,
                section_path=section.heading_path,
                position=position,
                text=piece,
                embedding_text="\n".join(context),
                content_hash=chunk_hash,
                page_hash=page.content_hash,
            )
        )
    return chunks
