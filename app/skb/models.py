from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class SkbModule:
    """A user-facing SKB module and its DokuWiki root namespace."""

    namespace: str
    label: str
    start_page_id: str


# Keep this order aligned with the product navigation shown by SKB.  The
# namespace is the stable value used to classify every descendant page.
SKB_MODULES: tuple[SkbModule, ...] = (
    SkbModule("spay", "Payroll", "spay:spay"),
    SkbModule("shrm", "Human Resources", "shrm:shrm"),
    SkbModule("sgc", "Gestion Commerciale", "sgc:sgc"),
    SkbModule("sacc", "Accounting", "sacc:sacc"),
    SkbModule("sfar", "Fixed Asset", "sfar:sfar"),
    SkbModule("sef", "Equipment Follow-up", "sef:sef"),
    SkbModule("sim", "Incident Management", "sim:sim"),
    SkbModule("seam", "SEAM", "seam:seam"),
    SkbModule("pms", "PMS", "pms:pms"),
    SkbModule("sess", "SESS", "sess:sess"),
)

_MODULE_BY_NAMESPACE = {item.namespace.casefold(): item for item in SKB_MODULES}
_MODULE_BY_LABEL = {item.label.casefold(): item for item in SKB_MODULES}
_ALL_MODULE_VALUES = {
    "",
    "all",
    "all modules",
    "tous",
    "tous les modules",
    "tout",
    "*",
}


def module_for_page_id(page_id: str) -> str | None:
    """Return the canonical menu label for a DokuWiki page id."""

    namespace = page_id.strip().casefold().split(":", 1)[0]
    item = _MODULE_BY_NAMESPACE.get(namespace)
    return item.label if item else None


def normalize_module_filter(module: str | None) -> str | None:
    """Normalize either a menu label or namespace to the stored label.

    Unknown non-empty values are retained.  They remain safe SQL values (the
    vector store binds them as parameters) and simply match no rows unless an
    additional module was indexed deliberately.
    """

    if module is None:
        return None
    cleaned = " ".join(module.split()).strip()
    if cleaned.casefold() in _ALL_MODULE_VALUES:
        return None
    item = _MODULE_BY_LABEL.get(cleaned.casefold()) or _MODULE_BY_NAMESPACE.get(
        cleaned.casefold()
    )
    return item.label if item else cleaned


def content_digest(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class WikiPage:
    page_id: str
    title: str
    source_url: str
    module: str | None
    raw_text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class WikiSection:
    heading: str
    heading_path: tuple[str, ...]
    level: int
    text: str


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    page_id: str
    title: str
    source_url: str
    module: str | None
    section: str
    section_path: tuple[str, ...]
    position: int
    text: str
    embedding_text: str
    content_hash: str
    page_hash: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    page_id: str
    title: str
    source_url: str
    module: str | None
    section: str
    section_path: tuple[str, ...]
    text: str
    distance: float
    score: float


@dataclass(frozen=True, slots=True)
class PageUpsertResult:
    page_id: str
    chunks_upserted: int
    chunks_deleted: int


@dataclass(frozen=True, slots=True)
class DeleteResult:
    pages: int
    chunks: int


@dataclass(frozen=True, slots=True)
class StoreStats:
    pages: int
    chunks: int
    modules: int
    generation_id: str | None = None
    signature: str | None = None


@dataclass(frozen=True, slots=True)
class ActivationResult:
    activated: bool
    generation_id: str
    previous_generation_id: str | None
    reason: str
    missing_pages: int = 0


@dataclass(frozen=True, slots=True)
class ActiveGeneration:
    generation_id: str
    index_signature: str


@dataclass(slots=True)
class IndexStats:
    generation_id: str | None = None
    previous_generation_id: str | None = None
    discovered_pages: int = 0
    fetched_pages: int = 0
    unchanged_pages: int = 0
    copied_pages: int = 0
    copied_chunks: int = 0
    indexed_pages: int = 0
    failed_pages: int = 0
    chunks_total: int = 0
    embedded_chunks: int = 0
    upserted_chunks: int = 0
    deleted_chunks: int = 0
    deleted_pages: int = 0
    deletion_skipped: bool = False
    activated: bool = False
    activation_deferred: bool = False
    activation_reason: str | None = None
    errors: list[str] = field(default_factory=list)
