from __future__ import annotations

import os
from dataclasses import replace

import pytest

from app.skb.models import WikiPage
from app.skb.parser import chunk_page
from app.skb.vector_store import (
    IndexNotReadyError,
    IndexSyncLockedError,
    MariaDBVectorStore,
)


pytestmark = pytest.mark.skipif(
    not os.getenv("SKB_TEST_MARIADB_DATABASE"),
    reason="set SKB_TEST_MARIADB_DATABASE to run the MariaDB 11.8 integration test",
)


@pytest.mark.asyncio
async def test_native_vector_schema_upsert_search_and_cleanup():
    store = MariaDBVectorStore(
        host=os.getenv("SKB_TEST_MARIADB_HOST", "127.0.0.1"),
        port=int(os.getenv("SKB_TEST_MARIADB_PORT", "3306")),
        user=os.getenv("SKB_TEST_MARIADB_USER", "root"),
        password=os.getenv("SKB_TEST_MARIADB_PASSWORD", ""),
        database=os.environ["SKB_TEST_MARIADB_DATABASE"],
        dimension=3,
    )
    await store.initialize()
    try:
        page = WikiPage(
            page_id="spay:test:vector",
            title="Vector test",
            source_url="http://skb.uniconsults.mu/doku.php?id=spay:test:vector",
            module="Payroll",
            raw_text=(
                "====== Vector test ======\nPayroll vector integration content "
                "with enough detail to produce one useful semantic search chunk."
            ),
            content_hash="1" * 64,
        )
        chunks = chunk_page(page)
        assert len(chunks) == 1
        signature = "a" * 64
        generation = await store.begin_generation(signature, 1)
        result = await store.upsert_page(
            generation,
            page,
            chunks,
            [[1.0, 0.0, 0.0]],
        )
        assert result.chunks_upserted == 1

        with pytest.raises(IndexNotReadyError):
            await store.search(
                [1.0, 0.0, 0.0],
                module="spay",
                expected_index_signature=signature,
            )

        activated = await store.activate_generation(generation, [page.page_id])
        assert activated.activated

        matches = await store.search(
            [1.0, 0.0, 0.0],
            module="spay",
            limit=3,
            max_distance=0.01,
            expected_index_signature=signature,
        )
        assert len(matches) == 1
        assert matches[0].page_id == page.page_id
        assert matches[0].source_url == page.source_url
        assert matches[0].distance == pytest.approx(0.0)

        stats = await store.stats()
        assert (stats.pages, stats.chunks, stats.modules) == (1, 1, 1)
        assert stats.generation_id == generation
        assert stats.signature == signature

        with pytest.raises(IndexNotReadyError):
            await store.search(
                [1.0, 0.0, 0.0],
                expected_index_signature="b" * 64,
            )

        # A complete second embedding space remains private until the pointer is
        # atomically switched.
        second_signature = "b" * 64
        second_generation = await store.begin_generation(second_signature, 2)
        updated_page = replace(
            page,
            raw_text=page.raw_text + " Updated generation.",
            content_hash="2" * 64,
        )
        updated_chunks = chunk_page(updated_page)
        await store.upsert_page(
            second_generation,
            updated_page,
            updated_chunks,
            [[0.0, 1.0, 0.0]],
        )
        second_page = replace(
            page,
            page_id="spay:test:second",
            title="Second vector test",
            source_url="http://skb.uniconsults.mu/doku.php?id=spay:test:second",
            raw_text=page.raw_text + " A second indexed page with useful details.",
            content_hash="3" * 64,
        )
        second_chunks = chunk_page(second_page)
        await store.upsert_page(
            second_generation,
            second_page,
            second_chunks,
            [[0.0, 0.0, 1.0]],
        )
        old_matches = await store.search(
            [1.0, 0.0, 0.0],
            max_distance=0.01,
            expected_index_signature=signature,
        )
        assert [match.page_id for match in old_matches] == [page.page_id]
        with pytest.raises(IndexNotReadyError):
            await store.search(
                [0.0, 1.0, 0.0],
                expected_index_signature=second_signature,
            )

        second_activation = await store.activate_generation(
            second_generation,
            [updated_page.page_id, second_page.page_id],
        )
        assert second_activation.activated
        assert (await store.stats()).pages == 2

        # A page-removal snapshot needs two identical successful crawls. The
        # active two-page generation remains searchable after the first attempt.
        third_generation = await store.begin_generation(second_signature, 1)
        await store.copy_page(
            second_generation,
            third_generation,
            updated_page.page_id,
        )
        deferred = await store.activate_generation(
            third_generation,
            [updated_page.page_id],
            minimum_retention_ratio=0.4,
        )
        assert not deferred.activated
        assert deferred.reason == "missing_pages_snapshot_requires_confirmation"
        assert (await store.stats()).pages == 2
        await store.discard_generation(third_generation)

        fourth_generation = await store.begin_generation(second_signature, 1)
        await store.copy_page(
            second_generation,
            fourth_generation,
            updated_page.page_id,
        )
        confirmed = await store.activate_generation(
            fourth_generation,
            [updated_page.page_id],
            minimum_retention_ratio=0.4,
        )
        assert confirmed.activated
        assert confirmed.previous_generation_id == second_generation
        assert confirmed.missing_pages == 1
        assert (await store.stats()).pages == 1

        # The active chunk is an exact clone of one that remains in the
        # superseded generation.  MariaDB's global ANN index can select the
        # superseded duplicate before applying the generation filter, leaving
        # no active result.  Search must rank only rows from the active
        # generation.
        cloned_matches = await store.search(
            [0.0, 1.0, 0.0],
            module="spay",
            limit=1,
            max_distance=0.01,
            expected_index_signature=second_signature,
        )
        assert len(cloned_matches) == 1
        assert cloned_matches[0].page_id == updated_page.page_id
        assert cloned_matches[0].source_url == updated_page.source_url
        assert cloned_matches[0].distance == pytest.approx(0.0)

        async with store.index_sync_lock():
            with pytest.raises(IndexSyncLockedError):
                async with store.index_sync_lock():
                    pass
    finally:
        await store.close()
