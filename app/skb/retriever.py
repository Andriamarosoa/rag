from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse
from uuid import UUID

from app.skb.embeddings import OllamaEmbeddingClient
from app.skb.models import RetrievedChunk, normalize_module_filter
from app.skb.vector_store import IndexNotReadyError, MariaDBVectorStore


class SkbRetriever:
    def __init__(
        self,
        embeddings: OllamaEmbeddingClient,
        store: MariaDBVectorStore,
        *,
        top_k: int = 6,
        max_distance: float = 0.45,
        index_signature: str | None = None,
        source_base_url: str = "http://skb.uniconsults.mu/",
        allowed_source_hosts: Iterable[str] | None = None,
    ) -> None:
        if embeddings.dimension != store.dimension:
            raise ValueError("embedding client and vector store dimensions must match")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not (0.0 <= max_distance <= 2.0):
            raise ValueError("max_distance must be between 0 and 2")
        parsed = urlparse(source_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("source_base_url must be an absolute HTTP(S) URL")

        self.embeddings = embeddings
        self.store = store
        self.top_k = min(int(top_k), 100)
        self.max_distance = float(max_distance)
        self.index_signature = index_signature
        self.allowed_source_hosts = frozenset(
            value.strip().rstrip(".").casefold()
            for value in (allowed_source_hosts or (parsed.hostname,))
            if value.strip()
        )

    def _trusted_source(self, result: RetrievedChunk) -> bool:
        if result.source_kind == "file":
            try:
                document_id = str(UUID(str(result.document_id)))
            except (TypeError, ValueError, AttributeError):
                return False
            return result.source_url == f"/knowledge/files/{document_id}/download"
        parsed = urlparse(result.source_url)
        return (
            parsed.scheme in {"http", "https"}
            and parsed.username is None
            and parsed.password is None
            and (parsed.hostname or "").rstrip(".").casefold()
            in self.allowed_source_hosts
        )

    async def retrieve(
        self, query: str, module: str | None = None
    ) -> list[RetrievedChunk]:
        cleaned_query = " ".join(query.split()).strip()
        if not cleaned_query:
            return []
        vector = await self.embeddings.embed_query(cleaned_query)
        try:
            results = await self.store.search(
                vector,
                module=normalize_module_filter(module),
                limit=self.top_k,
                max_distance=self.max_distance,
                expected_index_signature=self.index_signature,
            )
        except IndexNotReadyError:
            # Uploaded documents remain searchable while a first crawler
            # generation is still being prepared.
            results = []
        file_search = getattr(self.store, "search_knowledge_files", None)
        if callable(file_search):
            results.extend(
                await file_search(
                    vector,
                    module=normalize_module_filter(module),
                    limit=self.top_k,
                    max_distance=self.max_distance,
                    expected_index_signature=self.index_signature,
                )
            )
        trusted = [result for result in results if self._trusted_source(result)]
        trusted.sort(key=lambda result: result.distance)
        return trusted[: self.top_k]
