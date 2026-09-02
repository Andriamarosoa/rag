from __future__ import annotations

import asyncio
from dataclasses import replace

from app.skb.dokuwiki import DokuWikiClient
from app.skb.embeddings import OllamaEmbeddingClient
from app.skb.models import IndexStats, content_digest
from app.skb.parser import chunk_page
from app.skb.vector_store import MariaDBVectorStore


class SkbIndexer:
    """Build a complete private generation, then publish it atomically."""

    INDEX_FORMAT_VERSION = "2"

    def __init__(
        self,
        client: DokuWikiClient,
        embeddings: OllamaEmbeddingClient,
        store: MariaDBVectorStore,
        *,
        chunk_size: int = 1_600,
        chunk_overlap: int = 200,
        min_chunk_size: int = 80,
        fetch_batch_size: int = 64,
    ) -> None:
        if embeddings.dimension != store.dimension:
            raise ValueError("embedding client and vector store dimensions must match")
        if fetch_batch_size < 1:
            raise ValueError("fetch_batch_size must be positive")
        self.client = client
        self.embeddings = embeddings
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.fetch_batch_size = int(fetch_batch_size)
        self._sync_lock = asyncio.Lock()
        self.index_signature = content_digest(
            "skb-index",
            self.INDEX_FORMAT_VERSION,
            embeddings.model,
            str(embeddings.dimension),
            str(chunk_size),
            str(chunk_overlap),
            str(min_chunk_size),
        )

    @staticmethod
    def _record_error(stats: IndexStats, page_id: str, error: BaseException) -> None:
        if len(stats.errors) < 100:
            stats.errors.append(f"{page_id}: {type(error).__name__}: {error}")

    async def sync(self) -> IndexStats:
        async with self._sync_lock:
            await self.store.initialize()
            async with self.store.index_sync_lock():
                page_ids = await self.client.discover_page_ids()
                if not page_ids:
                    raise RuntimeError(
                        "DokuWiki discovery returned no pages; refusing empty sync"
                    )

                stats = IndexStats(discovered_pages=len(page_ids))
                active = await self.store.get_active_generation()
                if active is not None:
                    stats.previous_generation_id = active.generation_id
                generation_id = await self.store.begin_generation(
                    self.index_signature,
                    len(page_ids),
                )
                stats.generation_id = generation_id
                published = False

                try:
                    known_hashes = {}
                    if (
                        active is not None
                        and active.index_signature == self.index_signature
                    ):
                        known_hashes = await self.store.get_page_hashes(
                            page_ids,
                            generation_id=active.generation_id,
                        )

                    for start in range(0, len(page_ids), self.fetch_batch_size):
                        batch_ids = page_ids[start : start + self.fetch_batch_size]
                        fetched = await asyncio.gather(
                            *(self.client.fetch_page(page_id) for page_id in batch_ids),
                            return_exceptions=True,
                        )
                        for page_id, result in zip(batch_ids, fetched, strict=True):
                            if isinstance(result, BaseException) and not isinstance(
                                result, Exception
                            ):
                                raise result
                            if isinstance(result, Exception):
                                stats.failed_pages += 1
                                self._record_error(stats, page_id, result)
                                continue

                            page = replace(
                                result,
                                content_hash=content_digest(
                                    result.content_hash,
                                    self.index_signature,
                                ),
                            )
                            stats.fetched_pages += 1

                            if known_hashes.get(page.page_id) == page.content_hash:
                                try:
                                    copied = await self.store.copy_page(
                                        active.generation_id,
                                        generation_id,
                                        page.page_id,
                                    )
                                except Exception as exc:
                                    stats.failed_pages += 1
                                    self._record_error(stats, page.page_id, exc)
                                    continue
                                stats.unchanged_pages += 1
                                stats.copied_pages += 1
                                stats.copied_chunks += copied.chunks_upserted
                                stats.chunks_total += copied.chunks_upserted
                                continue

                            try:
                                chunks = chunk_page(
                                    page,
                                    chunk_size=self.chunk_size,
                                    chunk_overlap=self.chunk_overlap,
                                    min_chunk_size=self.min_chunk_size,
                                )
                                vectors = await self.embeddings.embed_texts(
                                    [chunk.embedding_text for chunk in chunks]
                                )
                                upsert = await self.store.upsert_page(
                                    generation_id,
                                    page,
                                    chunks,
                                    vectors,
                                )
                            except Exception as exc:
                                stats.failed_pages += 1
                                self._record_error(stats, page.page_id, exc)
                                continue

                            stats.indexed_pages += 1
                            stats.chunks_total += len(chunks)
                            stats.embedded_chunks += len(vectors)
                            stats.upserted_chunks += upsert.chunks_upserted
                            stats.deleted_chunks += upsert.chunks_deleted

                    if (
                        stats.failed_pages
                        or stats.fetched_pages != stats.discovered_pages
                    ):
                        stats.deletion_skipped = True
                        stats.activation_reason = "partial_sync_failed"
                        await self.store.discard_generation(
                            generation_id,
                            failed_pages=stats.failed_pages,
                        )
                        return stats

                    activation = await self.store.activate_generation(
                        generation_id,
                        page_ids,
                    )
                    stats.activated = activation.activated
                    stats.activation_reason = activation.reason
                    stats.deleted_pages = activation.missing_pages
                    if not activation.activated:
                        stats.activation_deferred = True
                        stats.deletion_skipped = True
                        await self.store.discard_generation(generation_id)
                        return stats

                    published = True
                    await self.store.prune_generations(keep=2)
                    return stats
                except BaseException:
                    if not published:
                        try:
                            await asyncio.shield(
                                self.store.discard_generation(
                                    generation_id,
                                    failed_pages=stats.failed_pages,
                                )
                            )
                        except Exception:
                            pass
                    raise
