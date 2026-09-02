from __future__ import annotations

import asyncio
import importlib
import json
import math
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Any, TypeVar
from uuid import uuid4

from app.skb.models import (
    ActivationResult,
    ActiveGeneration,
    DocumentChunk,
    PageUpsertResult,
    RetrievedChunk,
    StoreStats,
    WikiPage,
    normalize_module_filter,
)


class VectorStoreError(RuntimeError):
    pass


class IndexNotReadyError(VectorStoreError):
    pass


class IndexSyncLockedError(VectorStoreError):
    pass


_T = TypeVar("_T")
_RETRYABLE_TRANSACTION_CODES = frozenset({1020, 1205, 1213})


def _vector_json(values: Sequence[float], dimension: int) -> str:
    if len(values) != dimension:
        raise ValueError(f"embedding dimension is {len(values)}; expected {dimension}")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding contains a non-finite value")
    return json.dumps(vector, ensure_ascii=True, separators=(",", ":"))


def _error_code(error: BaseException) -> int | None:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if current.args and isinstance(current.args[0], int):
            return int(current.args[0])
        current = current.__cause__ or current.__context__
    return None


class MariaDBVectorStore:
    """Atomic, generation-based vector storage on MariaDB 11.8 LTS.

    A complete index is written into staging tables under a new generation id.
    Search reads only the generation referenced by the singleton state row, and
    activation changes that pointer in one transaction.  Consequently neither a
    first-time partial crawl nor embeddings from two models can be served.
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        *,
        dimension: int = 1_024,
        min_pool_size: int = 1,
        max_pool_size: int = 5,
        connect_timeout: float = 10.0,
        transaction_retry_attempts: int = 3,
        transaction_retry_backoff_seconds: float = 0.1,
    ) -> None:
        if not host.strip() or not user.strip() or not database.strip():
            raise ValueError("host, user, and database are required")
        if not (1 <= int(port) <= 65_535):
            raise ValueError("port must be between 1 and 65535")
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if min_pool_size < 1 or max_pool_size < min_pool_size:
            raise ValueError("invalid MariaDB pool sizes")
        if transaction_retry_attempts < 0:
            raise ValueError("transaction_retry_attempts cannot be negative")

        self.host = host.strip()
        self.port = int(port)
        self.user = user
        self.password = password
        self.database = database
        self.dimension = int(dimension)
        self.min_pool_size = int(min_pool_size)
        self.max_pool_size = int(max_pool_size)
        self.connect_timeout = max(0.1, float(connect_timeout))
        self.transaction_retry_attempts = int(transaction_retry_attempts)
        self.transaction_retry_backoff_seconds = max(
            0.0, float(transaction_retry_backoff_seconds)
        )
        lock_suffix = sha256(
            f"{self.host}:{self.port}/{self.database}".encode("utf-8")
        ).hexdigest()[:32]
        self._lock_name = f"skb-index-sync:{lock_suffix}"
        self._pool: Any | None = None
        self._aiomysql: Any | None = None
        self._initialize_lock = asyncio.Lock()

    async def __aenter__(self) -> MariaDBVectorStore:
        await self.initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise VectorStoreError("MariaDBVectorStore.initialize() has not been called")
        return self._pool

    async def initialize(self) -> None:
        if self._pool is not None:
            return
        async with self._initialize_lock:
            if self._pool is not None:
                return
            try:
                aiomysql = importlib.import_module("aiomysql")
            except ImportError as exc:
                raise VectorStoreError(
                    "aiomysql is required for MariaDBVectorStore"
                ) from exc

            pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                minsize=self.min_pool_size,
                maxsize=self.max_pool_size,
                connect_timeout=self.connect_timeout,
                charset="utf8mb4",
                autocommit=False,
                pool_recycle=3_600,
            )
            self._aiomysql = aiomysql
            self._pool = pool
            try:
                await self._initialize_schema()
            except BaseException:
                self._pool = None
                pool.close()
                await pool.wait_closed()
                raise

    async def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()
            await pool.wait_closed()

    async def _initialize_schema(self) -> None:
        pool = self._require_pool()
        aiomysql = self._aiomysql
        async with pool.acquire() as connection:
            async with connection.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("SELECT VERSION() AS version")
                row = await cursor.fetchone()
                version = str((row or {}).get("version", ""))
                match = re.search(r"(\d+)\.(\d+)", version)
                if not match or (int(match.group(1)), int(match.group(2))) < (11, 8):
                    raise VectorStoreError(
                        "MariaDB 11.8 LTS or newer is required for production "
                        f"VECTOR support; server is {version!r}"
                    )

                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS skb_index_generations (
                        generation_id CHAR(36) CHARACTER SET ascii
                            COLLATE ascii_bin NOT NULL PRIMARY KEY,
                        index_signature CHAR(64) CHARACTER SET ascii
                            COLLATE ascii_bin NOT NULL,
                        status VARCHAR(16) NOT NULL DEFAULT 'building',
                        discovered_pages INT UNSIGNED NOT NULL,
                        failed_pages INT UNSIGNED NOT NULL DEFAULT 0,
                        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                        completed_at TIMESTAMP(6) NULL,
                        KEY idx_skb_generations_status_created (status, created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        COLLATE=utf8mb4_unicode_ci
                    """
                )
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS skb_index_state (
                        singleton TINYINT UNSIGNED NOT NULL PRIMARY KEY,
                        active_generation_id CHAR(36) CHARACTER SET ascii
                            COLLATE ascii_bin NULL,
                        candidate_snapshot_hash CHAR(64) CHARACTER SET ascii
                            COLLATE ascii_bin NULL,
                        candidate_snapshot_count INT UNSIGNED NULL,
                        updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                            ON UPDATE CURRENT_TIMESTAMP(6),
                        CONSTRAINT chk_skb_index_state_singleton CHECK (singleton=1),
                        CONSTRAINT fk_skb_state_active_generation
                            FOREIGN KEY (active_generation_id)
                            REFERENCES skb_index_generations(generation_id)
                            ON DELETE SET NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        COLLATE=utf8mb4_unicode_ci
                    """
                )
                await cursor.execute(
                    "INSERT IGNORE INTO skb_index_state (singleton) VALUES (1)"
                )
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS skb_pages_v2 (
                        generation_id CHAR(36) CHARACTER SET ascii
                            COLLATE ascii_bin NOT NULL,
                        page_id VARCHAR(512) NOT NULL,
                        title VARCHAR(512) NOT NULL,
                        source_url VARCHAR(2048) NOT NULL,
                        module VARCHAR(128) NULL,
                        content_hash CHAR(64) CHARACTER SET ascii
                            COLLATE ascii_bin NOT NULL,
                        indexed_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                        PRIMARY KEY (generation_id, page_id),
                        KEY idx_skb_pages_v2_page (page_id),
                        KEY idx_skb_pages_v2_module (generation_id, module),
                        CONSTRAINT fk_skb_pages_v2_generation
                            FOREIGN KEY (generation_id)
                            REFERENCES skb_index_generations(generation_id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        COLLATE=utf8mb4_unicode_ci
                    """
                )
                await cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS skb_chunks_v2 (
                        row_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        generation_id CHAR(36) CHARACTER SET ascii
                            COLLATE ascii_bin NOT NULL,
                        chunk_id CHAR(64) CHARACTER SET ascii
                            COLLATE ascii_bin NOT NULL,
                        page_id VARCHAR(512) NOT NULL,
                        title VARCHAR(512) NOT NULL,
                        source_url VARCHAR(2048) NOT NULL,
                        module VARCHAR(128) NULL,
                        section VARCHAR(512) NOT NULL,
                        section_path TEXT NOT NULL,
                        position INT UNSIGNED NOT NULL,
                        content MEDIUMTEXT NOT NULL,
                        content_hash CHAR(64) CHARACTER SET ascii
                            COLLATE ascii_bin NOT NULL,
                        page_hash CHAR(64) CHARACTER SET ascii
                            COLLATE ascii_bin NOT NULL,
                        embedding VECTOR({self.dimension}) NOT NULL,
                        indexed_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                        UNIQUE KEY uq_skb_chunks_v2_generation_chunk
                            (generation_id, chunk_id),
                        KEY idx_skb_chunks_v2_page_position
                            (generation_id, page_id, position),
                        KEY idx_skb_chunks_v2_module (generation_id, module),
                        VECTOR INDEX skb_chunks_v2_embedding_idx (embedding)
                            M=8 DISTANCE=cosine,
                        CONSTRAINT fk_skb_chunks_v2_page
                            FOREIGN KEY (generation_id, page_id)
                            REFERENCES skb_pages_v2(generation_id, page_id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        COLLATE=utf8mb4_unicode_ci
                    """
                )
                await cursor.execute(
                    """
                    SELECT COLUMN_TYPE AS column_type
                    FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA=%s AND TABLE_NAME='skb_chunks_v2'
                        AND COLUMN_NAME='embedding'
                    """,
                    (self.database,),
                )
                column = await cursor.fetchone()
                column_type = str((column or {}).get("column_type", ""))
                dimension_match = re.fullmatch(r"vector\((\d+)\)", column_type, re.I)
                if not dimension_match or int(dimension_match.group(1)) != self.dimension:
                    raise VectorStoreError(
                        "existing skb_chunks_v2 embedding dimension does not match "
                        f"the configured dimension {self.dimension}: {column_type!r}"
                    )
            await connection.commit()

    def _is_retryable_transaction_error(self, error: BaseException) -> bool:
        return _error_code(error) in _RETRYABLE_TRANSACTION_CODES

    async def _run_transaction(
        self, operation: Callable[[Any], Awaitable[_T]]
    ) -> _T:
        pool = self._require_pool()
        for attempt in range(self.transaction_retry_attempts + 1):
            async with pool.acquire() as connection:
                try:
                    result = await operation(connection)
                    await connection.commit()
                    return result
                except BaseException as exc:
                    await connection.rollback()
                    retryable = isinstance(exc, Exception) and self._is_retryable_transaction_error(exc)
                    if retryable and attempt < self.transaction_retry_attempts:
                        await asyncio.sleep(
                            self.transaction_retry_backoff_seconds * (2**attempt)
                        )
                        continue
                    raise
        raise AssertionError("transaction retry loop did not return")

    @asynccontextmanager
    async def index_sync_lock(self, timeout_seconds: int = 0) -> AsyncIterator[None]:
        """Hold a MariaDB advisory lock on a dedicated connection."""

        if self._aiomysql is None:
            raise VectorStoreError("MariaDBVectorStore.initialize() has not been called")
        connection = await self._aiomysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            db=self.database,
            connect_timeout=self.connect_timeout,
            charset="utf8mb4",
            autocommit=True,
        )
        acquired = False
        try:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT GET_LOCK(%s, %s) AS acquired",
                    (self._lock_name, max(0, int(timeout_seconds))),
                )
                row = await cursor.fetchone()
                acquired = int((row or {}).get("acquired") or 0) == 1
            if not acquired:
                raise IndexSyncLockedError("another SKB index synchronization is running")
            yield
        finally:
            if acquired:
                try:
                    async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                        await cursor.execute(
                            "SELECT RELEASE_LOCK(%s) AS released", (self._lock_name,)
                        )
                except Exception:
                    pass
            connection.close()

    async def begin_generation(
        self, index_signature: str, discovered_pages: int
    ) -> str:
        signature = index_signature.strip()
        if not signature or len(signature) > 64:
            raise ValueError("index_signature must contain at most 64 characters")
        if discovered_pages < 1:
            raise ValueError("discovered_pages must be positive")
        generation_id = str(uuid4())

        async def operation(connection: Any) -> str:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    INSERT INTO skb_index_generations
                        (generation_id, index_signature, status, discovered_pages)
                    VALUES (%s, %s, 'building', %s)
                    """,
                    (generation_id, signature, int(discovered_pages)),
                )
            return generation_id

        return await self._run_transaction(operation)

    async def get_active_generation(self) -> ActiveGeneration | None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    SELECT g.generation_id, g.index_signature
                    FROM skb_index_state AS s
                    LEFT JOIN skb_index_generations AS g
                        ON g.generation_id=s.active_generation_id
                    WHERE s.singleton=1
                    """
                )
                row = await cursor.fetchone()
        if not row or not row.get("generation_id"):
            return None
        return ActiveGeneration(
            generation_id=str(row["generation_id"]),
            index_signature=str(row["index_signature"]),
        )

    async def get_page_hash(self, page_id: str) -> str | None:
        hashes = await self.get_page_hashes([page_id])
        return hashes.get(page_id)

    async def get_page_hashes(
        self,
        page_ids: Sequence[str],
        *,
        generation_id: str | None = None,
    ) -> dict[str, str]:
        if not page_ids:
            return {}
        if generation_id is None:
            active = await self.get_active_generation()
            if active is None:
                return {}
            generation_id = active.generation_id

        pool = self._require_pool()
        output: dict[str, str] = {}
        async with pool.acquire() as connection:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                for start in range(0, len(page_ids), 1_000):
                    batch = list(page_ids[start : start + 1_000])
                    placeholders = ",".join(["%s"] * len(batch))
                    await cursor.execute(
                        "SELECT page_id, content_hash FROM skb_pages_v2 "
                        f"WHERE generation_id=%s AND page_id IN ({placeholders})",
                        (generation_id, *batch),
                    )
                    for row in await cursor.fetchall():
                        output[str(row["page_id"])] = str(row["content_hash"])
        return output

    async def copy_page(
        self,
        source_generation_id: str,
        target_generation_id: str,
        page_id: str,
    ) -> PageUpsertResult:
        async def operation(connection: Any) -> PageUpsertResult:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    INSERT INTO skb_pages_v2
                        (generation_id, page_id, title, source_url, module,
                         content_hash, indexed_at)
                    SELECT %s, page_id, title, source_url, module, content_hash,
                           indexed_at
                    FROM skb_pages_v2
                    WHERE generation_id=%s AND page_id=%s
                    """,
                    (target_generation_id, source_generation_id, page_id),
                )
                if int(cursor.rowcount) != 1:
                    raise VectorStoreError(
                        f"cannot copy missing page {page_id!r} from active generation"
                    )
                await cursor.execute(
                    """
                    INSERT INTO skb_chunks_v2
                        (generation_id, chunk_id, page_id, title, source_url,
                         module, section, section_path, position, content,
                         content_hash, page_hash, embedding, indexed_at)
                    SELECT %s, chunk_id, page_id, title, source_url, module,
                           section, section_path, position, content, content_hash,
                           page_hash, embedding, indexed_at
                    FROM skb_chunks_v2
                    WHERE generation_id=%s AND page_id=%s
                    """,
                    (target_generation_id, source_generation_id, page_id),
                )
                chunks_copied = max(0, int(cursor.rowcount))
            return PageUpsertResult(page_id, chunks_copied, 0)

        return await self._run_transaction(operation)

    async def upsert_page(
        self,
        generation_id: str,
        page: WikiPage,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> PageUpsertResult:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if any(chunk.page_id != page.page_id for chunk in chunks):
            raise ValueError("all chunks must belong to the supplied page")
        vector_payloads = [_vector_json(vector, self.dimension) for vector in embeddings]

        async def operation(connection: Any) -> PageUpsertResult:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    INSERT INTO skb_pages_v2
                        (generation_id, page_id, title, source_url, module,
                         content_hash)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        title=VALUES(title), source_url=VALUES(source_url),
                        module=VALUES(module), content_hash=VALUES(content_hash),
                        indexed_at=CURRENT_TIMESTAMP(6)
                    """,
                    (
                        generation_id,
                        page.page_id,
                        page.title,
                        page.source_url,
                        page.module,
                        page.content_hash,
                    ),
                )
                if chunks:
                    rows = [
                        (
                            generation_id,
                            chunk.chunk_id,
                            chunk.page_id,
                            chunk.title,
                            chunk.source_url,
                            chunk.module,
                            chunk.section,
                            json.dumps(
                                chunk.section_path,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            chunk.position,
                            chunk.text,
                            chunk.content_hash,
                            chunk.page_hash,
                            vector_payload,
                        )
                        for chunk, vector_payload in zip(
                            chunks, vector_payloads, strict=True
                        )
                    ]
                    await cursor.executemany(
                        """
                        INSERT INTO skb_chunks_v2
                            (generation_id, chunk_id, page_id, title, source_url,
                             module, section, section_path, position, content,
                             content_hash, page_hash, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, VEC_FromText(%s))
                        ON DUPLICATE KEY UPDATE
                            page_id=VALUES(page_id), title=VALUES(title),
                            source_url=VALUES(source_url), module=VALUES(module),
                            section=VALUES(section), section_path=VALUES(section_path),
                            position=VALUES(position), content=VALUES(content),
                            content_hash=VALUES(content_hash),
                            page_hash=VALUES(page_hash), embedding=VALUES(embedding),
                            indexed_at=CURRENT_TIMESTAMP(6)
                        """,
                        rows,
                    )
                await cursor.execute(
                    """
                    DELETE FROM skb_chunks_v2
                    WHERE generation_id=%s AND page_id=%s AND page_hash<>%s
                    """,
                    (generation_id, page.page_id, page.content_hash),
                )
                deleted = max(0, int(cursor.rowcount))
            return PageUpsertResult(page.page_id, len(chunks), deleted)

        return await self._run_transaction(operation)

    @staticmethod
    def _snapshot_hash(page_ids: Sequence[str]) -> str:
        digest = sha256()
        for page_id in sorted(set(page_ids)):
            digest.update(page_id.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    async def activate_generation(
        self,
        generation_id: str,
        page_ids: Sequence[str],
        *,
        minimum_retention_ratio: float = 0.90,
    ) -> ActivationResult:
        unique_page_ids = sorted(set(page_ids))
        if not unique_page_ids:
            raise ValueError("cannot activate an empty generation")
        if not (0.0 < minimum_retention_ratio <= 1.0):
            raise ValueError("minimum_retention_ratio must be in (0, 1]")
        snapshot_hash = self._snapshot_hash(unique_page_ids)

        async def operation(connection: Any) -> ActivationResult:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    SELECT status, discovered_pages
                    FROM skb_index_generations
                    WHERE generation_id=%s
                    FOR UPDATE
                    """,
                    (generation_id,),
                )
                generation = await cursor.fetchone()
                if not generation:
                    raise VectorStoreError(f"unknown generation {generation_id!r}")
                await cursor.execute(
                    "SELECT COUNT(*) AS count FROM skb_pages_v2 WHERE generation_id=%s",
                    (generation_id,),
                )
                staged_count = int((await cursor.fetchone())["count"])
                if staged_count != len(unique_page_ids):
                    raise VectorStoreError(
                        "staging generation is incomplete: "
                        f"{staged_count}/{len(unique_page_ids)} pages"
                    )

                await cursor.execute(
                    """
                    SELECT active_generation_id, candidate_snapshot_hash,
                           candidate_snapshot_count
                    FROM skb_index_state
                    WHERE singleton=1
                    FOR UPDATE
                    """
                )
                state = await cursor.fetchone() or {}
                previous_id = (
                    str(state["active_generation_id"])
                    if state.get("active_generation_id")
                    else None
                )
                active_count = 0
                missing_pages = 0

                if previous_id:
                    snapshot_table = (
                        "tmp_skb_snapshot_" + generation_id.replace("-", "")
                    )
                    await cursor.execute(
                        f"""
                        CREATE TEMPORARY TABLE {snapshot_table} (
                            page_id VARCHAR(512) NOT NULL PRIMARY KEY
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                            COLLATE=utf8mb4_unicode_ci
                        """
                    )
                    await cursor.executemany(
                        f"INSERT INTO {snapshot_table} (page_id) VALUES (%s)",
                        [(page_id,) for page_id in unique_page_ids],
                    )
                    await cursor.execute(
                        "SELECT COUNT(*) AS count FROM skb_pages_v2 WHERE generation_id=%s",
                        (previous_id,),
                    )
                    active_count = int((await cursor.fetchone())["count"])
                    await cursor.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM skb_pages_v2 AS p
                        LEFT JOIN {snapshot_table} AS live
                            ON live.page_id=p.page_id
                        WHERE p.generation_id=%s AND live.page_id IS NULL
                        """.format(snapshot_table=snapshot_table),
                        (previous_id,),
                    )
                    missing_pages = int((await cursor.fetchone())["count"])
                    await cursor.execute(
                        f"DROP TEMPORARY TABLE {snapshot_table}"
                    )

                if previous_id and missing_pages:
                    ratio = len(unique_page_ids) / max(1, active_count)
                    confirmed = (
                        state.get("candidate_snapshot_hash") == snapshot_hash
                        and int(state.get("candidate_snapshot_count") or -1)
                        == len(unique_page_ids)
                    )
                    ratio_safe = ratio >= minimum_retention_ratio
                    await cursor.execute(
                        """
                        UPDATE skb_index_state
                        SET candidate_snapshot_hash=%s,
                            candidate_snapshot_count=%s
                        WHERE singleton=1
                        """,
                        (snapshot_hash, len(unique_page_ids)),
                    )
                    if not confirmed or not ratio_safe:
                        reason = (
                            "retention_ratio_below_safety_floor"
                            if not ratio_safe
                            else "missing_pages_snapshot_requires_confirmation"
                        )
                        await cursor.execute(
                            """
                            UPDATE skb_index_generations
                            SET status='deferred', completed_at=CURRENT_TIMESTAMP(6)
                            WHERE generation_id=%s
                            """,
                            (generation_id,),
                        )
                        return ActivationResult(
                            activated=False,
                            generation_id=generation_id,
                            previous_generation_id=previous_id,
                            reason=reason,
                            missing_pages=missing_pages,
                        )

                await cursor.execute(
                    """
                    UPDATE skb_index_generations
                    SET status='superseded'
                    WHERE generation_id=%s AND status='active'
                    """,
                    (previous_id,),
                )
                await cursor.execute(
                    """
                    UPDATE skb_index_generations
                    SET status='active', completed_at=CURRENT_TIMESTAMP(6)
                    WHERE generation_id=%s
                    """,
                    (generation_id,),
                )
                await cursor.execute(
                    """
                    UPDATE skb_index_state
                    SET active_generation_id=%s,
                        candidate_snapshot_hash=NULL,
                        candidate_snapshot_count=NULL
                    WHERE singleton=1
                    """,
                    (generation_id,),
                )
            return ActivationResult(
                activated=True,
                generation_id=generation_id,
                previous_generation_id=previous_id,
                reason="activated",
                missing_pages=missing_pages,
            )

        return await self._run_transaction(operation)

    async def discard_generation(
        self, generation_id: str, *, failed_pages: int = 0
    ) -> None:
        async def operation(connection: Any) -> None:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    UPDATE skb_index_generations
                    SET failed_pages=%s, status='failed',
                        completed_at=CURRENT_TIMESTAMP(6)
                    WHERE generation_id=%s AND status<>'active'
                    """,
                    (max(0, int(failed_pages)), generation_id),
                )
                await cursor.execute(
                    """
                    DELETE FROM skb_index_generations
                    WHERE generation_id=%s AND status='failed'
                    """,
                    (generation_id,),
                )

        await self._run_transaction(operation)

    async def prune_generations(self, keep: int = 2) -> int:
        """Keep the active generation plus the newest superseded generations."""

        keep = max(1, int(keep))

        async def operation(connection: Any) -> int:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    SELECT generation_id
                    FROM skb_index_generations
                    WHERE status<>'active'
                    ORDER BY created_at DESC
                    """
                )
                rows = await cursor.fetchall()
                stale_ids = [str(row["generation_id"]) for row in rows[keep - 1 :]]
                if stale_ids:
                    await cursor.executemany(
                        "DELETE FROM skb_index_generations WHERE generation_id=%s",
                        [(generation_id,) for generation_id in stale_ids],
                    )
                return len(stale_ids)

        return await self._run_transaction(operation)

    async def stats(self) -> StoreStats:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    SELECT
                        s.active_generation_id AS generation_id,
                        (SELECT g.index_signature
                         FROM skb_index_generations AS g
                         WHERE g.generation_id=s.active_generation_id) AS signature,
                        (SELECT COUNT(*)
                         FROM skb_pages_v2 AS p
                         WHERE p.generation_id=s.active_generation_id) AS pages,
                        (SELECT COUNT(*)
                         FROM skb_chunks_v2 AS c
                         WHERE c.generation_id=s.active_generation_id) AS chunks,
                        (SELECT COUNT(DISTINCT p.module)
                         FROM skb_pages_v2 AS p
                         WHERE p.generation_id=s.active_generation_id
                           AND p.module IS NOT NULL) AS modules
                    FROM skb_index_state AS s
                    WHERE s.singleton=1
                    """
                )
                row = await cursor.fetchone() or {"pages": 0, "chunks": 0, "modules": 0}
        return StoreStats(
            pages=int(row.get("pages") or 0),
            chunks=int(row.get("chunks") or 0),
            modules=int(row.get("modules") or 0),
            generation_id=(
                str(row["generation_id"]) if row.get("generation_id") else None
            ),
            signature=(str(row["signature"]) if row.get("signature") else None),
        )

    async def search(
        self,
        query_embedding: Sequence[float],
        module: str | None = None,
        limit: int = 6,
        max_distance: float | None = 0.45,
        *,
        expected_index_signature: str | None = None,
    ) -> list[RetrievedChunk]:
        vector = _vector_json(query_embedding, self.dimension)
        limit = max(1, min(int(limit), 100))
        if max_distance is not None and not (0.0 <= float(max_distance) <= 2.0):
            raise ValueError("max_distance must be between 0 and 2")
        canonical_module = normalize_module_filter(module)

        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    SELECT g.generation_id, g.index_signature
                    FROM skb_index_state AS s
                    LEFT JOIN skb_index_generations AS g
                        ON g.generation_id=s.active_generation_id
                    WHERE s.singleton=1
                    """
                )
                active = await cursor.fetchone()
                if not active or not active.get("generation_id"):
                    raise IndexNotReadyError("no complete SKB index generation is active")
                active_signature = str(active["index_signature"])
                if (
                    expected_index_signature is not None
                    and active_signature != expected_index_signature
                ):
                    raise IndexNotReadyError(
                        "the active SKB index uses a different embedding configuration"
                    )

                where_parts = ["c.generation_id=%s"]
                parameters: list[Any] = [vector, str(active["generation_id"])]
                if canonical_module is not None:
                    where_parts.append("c.module=%s")
                    parameters.append(canonical_module)
                parameters.append(limit)
                where = " AND ".join(where_parts)
                sql = f"""
                    SELECT c.chunk_id, c.page_id, c.title, c.source_url,
                           c.module, c.section, c.section_path, c.content,
                           VEC_DISTANCE_COSINE(
                               c.embedding, VEC_FromText(%s)
                           ) AS distance
                    FROM skb_chunks_v2 AS c
                    WHERE {where}
                    ORDER BY distance ASC
                    LIMIT %s
                """
                await cursor.execute(sql, tuple(parameters))
                rows = await cursor.fetchall()

        results: list[RetrievedChunk] = []
        for row in rows:
            if row.get("distance") is None:
                continue
            distance = float(row["distance"])
            if max_distance is not None and distance > float(max_distance):
                continue
            try:
                raw_path = json.loads(str(row["section_path"]))
                section_path = tuple(str(item) for item in raw_path)
            except (TypeError, ValueError, json.JSONDecodeError):
                section_path = (str(row["section"]),)
            results.append(
                RetrievedChunk(
                    chunk_id=str(row["chunk_id"]),
                    page_id=str(row["page_id"]),
                    title=str(row["title"]),
                    source_url=str(row["source_url"]),
                    module=(
                        str(row["module"]) if row.get("module") is not None else None
                    ),
                    section=str(row["section"]),
                    section_path=section_path,
                    text=str(row["content"]),
                    distance=distance,
                    score=1.0 - distance,
                )
            )
        return results
