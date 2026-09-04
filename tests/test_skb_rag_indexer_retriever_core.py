from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from app.skb.indexer import SkbIndexer
from app.skb.models import (
    ActivationResult,
    ActiveGeneration,
    PageUpsertResult,
    RetrievedChunk,
    WikiPage,
    content_digest,
)
from app.skb.retriever import SkbRetriever


def _wiki_page(page_id: str, content_hash: str) -> WikiPage:
    return WikiPage(
        page_id=page_id,
        title=page_id,
        source_url=f"http://skb.uniconsults.mu/doku.php?id={page_id}",
        module="Payroll",
        raw_text=f"====== {page_id} ======\nUseful content for {page_id}.",
        content_hash=content_hash,
    )


class _FakeWiki:
    def __init__(self, pages, failures=()):
        self.pages = {page.page_id: page for page in pages}
        self.failures = set(failures)

    async def discover_page_ids(self):
        return sorted(set(self.pages) | self.failures)

    async def fetch_page(self, page_id):
        if page_id in self.failures:
            raise RuntimeError("upstream unavailable")
        return self.pages[page_id]


class _FakeEmbeddings:
    dimension = 3
    model = "fake-embedding"

    def __init__(self):
        self.embedded = []

    async def embed_texts(self, texts):
        self.embedded.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def embed_query(self, text):
        self.query = text
        return [1.0, 0.0, 0.0]


class _FakeStore:
    dimension = 3

    def __init__(self, hashes=None):
        self.hashes = hashes or {}
        self.active = None
        self.copies = []
        self.discarded = []
        self.upserts = []
        self.search_args = None
        self.activated_with = None

    async def initialize(self):
        pass

    @asynccontextmanager
    async def index_sync_lock(self):
        yield

    async def get_active_generation(self):
        return self.active

    async def begin_generation(self, _signature, _discovered_pages):
        return "generation-new"

    async def get_page_hashes(self, _page_ids, *, generation_id=None):
        return self.hashes

    async def copy_page(self, source_generation_id, target_generation_id, page_id):
        self.copies.append((source_generation_id, target_generation_id, page_id))
        return PageUpsertResult(page_id, 1, 0)

    async def upsert_page(self, generation_id, page, chunks, vectors):
        self.upserts.append((generation_id, page, chunks, vectors))
        return PageUpsertResult(page.page_id, len(chunks), 0)

    async def activate_generation(self, generation_id, page_ids):
        self.activated_with = (generation_id, list(page_ids))
        return ActivationResult(True, generation_id, "generation-old", "activated")

    async def discard_generation(self, generation_id, *, failed_pages=0):
        self.discarded.append((generation_id, failed_pages))

    async def prune_generations(self, keep=2):
        self.pruned_keep = keep
        return 0

    async def search(
        self,
        vector,
        module,
        limit,
        max_distance,
        *,
        expected_index_signature=None,
    ):
        self.search_args = (
            vector,
            module,
            limit,
            max_distance,
            expected_index_signature,
        )
        return self.results


@pytest.mark.asyncio
async def test_indexer_skips_unchanged_page_and_deletes_only_after_complete_sync():
    pages = [_wiki_page("spay:a", "a" * 64), _wiki_page("spay:b", "b" * 64)]
    store = _FakeStore()
    embeddings = _FakeEmbeddings()
    indexer = SkbIndexer(_FakeWiki(pages), embeddings, store)
    store.active = ActiveGeneration("generation-old", indexer.index_signature)
    store.hashes = {
        "spay:a": content_digest("a" * 64, indexer.index_signature),
    }

    stats = await indexer.sync()

    assert stats.unchanged_pages == 1
    assert stats.indexed_pages == 1
    assert stats.activated
    assert not stats.deletion_skipped
    assert store.activated_with == (
        "generation-new",
        ["spay:a", "spay:b"],
    )
    assert store.copies == [
        ("generation-old", "generation-new", "spay:a")
    ]
    assert [item[1].page_id for item in store.upserts] == ["spay:b"]


@pytest.mark.asyncio
async def test_indexer_never_deletes_after_a_partial_fetch_failure():
    store = _FakeStore()
    indexer = SkbIndexer(
        _FakeWiki([_wiki_page("spay:a", "a" * 64)], failures={"spay:b"}),
        _FakeEmbeddings(),
        store,
    )

    stats = await indexer.sync()

    assert stats.failed_pages == 1
    assert stats.deletion_skipped
    assert not stats.activated
    assert store.activated_with is None
    assert store.discarded == [("generation-new", 1)]


@pytest.mark.asyncio
async def test_retriever_canonicalizes_module_and_drops_untrusted_source_urls():
    trusted = RetrievedChunk(
        chunk_id="1",
        page_id="spay:a",
        title="A",
        source_url="http://skb.uniconsults.mu/doku.php?id=spay:a",
        module="Payroll",
        section="A",
        section_path=("A",),
        text="trusted",
        distance=0.1,
        score=0.9,
    )
    store = _FakeStore()
    store.results = [trusted, replace(trusted, chunk_id="2", source_url="https://evil.example/")]
    embeddings = _FakeEmbeddings()
    retriever = SkbRetriever(
        embeddings,
        store,
        top_k=4,
        index_signature="expected-index",
    )

    results = await retriever.retrieve("  leave   balance ", module="spay")

    assert results == [trusted]
    assert store.search_args == (
        [1.0, 0.0, 0.0],
        "Payroll",
        4,
        0.45,
        "expected-index",
    )
    assert embeddings.query == "leave balance"


@pytest.mark.asyncio
async def test_retriever_merges_uploaded_docx_chunks_with_skb_results():
    document_id = "248c6ee3-74e4-4d10-9439-024fd506f7d8"
    skb = RetrievedChunk(
        chunk_id="skb",
        page_id="spay:a",
        title="SKB",
        source_url="http://skb.uniconsults.mu/doku.php?id=spay:a",
        module="Payroll",
        section="A",
        section_path=("A",),
        text="skb",
        distance=0.2,
        score=0.8,
    )
    uploaded = replace(
        skb,
        chunk_id="docx",
        page_id=f"spay:file:{document_id}",
        source_url=f"/knowledge/files/{document_id}/download",
        text="docx",
        distance=0.1,
        score=0.9,
        source_kind="file",
        document_id=document_id,
    )

    class Store(_FakeStore):
        async def search_knowledge_files(self, *_args, **_kwargs):
            return [uploaded]

    store = Store()
    store.results = [skb]
    results = await SkbRetriever(
        _FakeEmbeddings(), store, top_k=2, index_signature="expected-index"
    ).retrieve("password", module="spay")

    assert [item.chunk_id for item in results] == ["docx", "skb"]
