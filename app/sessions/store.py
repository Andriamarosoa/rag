from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any
from uuid import uuid4

from .models import ChatMessage, ChatSession, utc_now


class SessionStore:
    """MariaDB persistence for chats and messages."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        min_pool_size: int = 1,
        max_pool_size: int = 5,
        connect_timeout: float = 10.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.database = database
        self.min_pool_size = int(min_pool_size)
        self.max_pool_size = int(max_pool_size)
        self.connect_timeout = float(connect_timeout)
        self._pool: Any | None = None
        self._aiomysql: Any | None = None
        self._initialize_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._pool is not None:
            return
        async with self._initialize_lock:
            if self._pool is not None:
                return
            aiomysql = importlib.import_module("aiomysql")
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

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("SessionStore.initialize() has not been called")
        return self._pool

    async def _initialize_schema(self) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chats (
                        id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin
                            NOT NULL PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        summary MEDIUMTEXT NOT NULL,
                        estimated_tokens BIGINT UNSIGNED NOT NULL DEFAULT 0,
                        codex_thread_id VARCHAR(255) NULL,
                        created_at VARCHAR(40) NOT NULL,
                        updated_at VARCHAR(40) NOT NULL,
                        KEY idx_chats_user (user_id, updated_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        COLLATE=utf8mb4_unicode_ci
                    """
                )
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin
                            NOT NULL PRIMARY KEY,
                        chat_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin
                            NOT NULL,
                        role VARCHAR(16) NOT NULL,
                        content MEDIUMTEXT NOT NULL,
                        metadata_json JSON NOT NULL,
                        created_at VARCHAR(40) NOT NULL,
                        KEY idx_messages_chat (chat_id, created_at),
                        CONSTRAINT fk_messages_chat FOREIGN KEY (chat_id)
                            REFERENCES chats(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        COLLATE=utf8mb4_unicode_ci
                    """
                )
            await connection.commit()

    async def get_or_create_chat(
        self, user_id: str, chat_id: str | None = None
    ) -> ChatSession:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                if chat_id:
                    await cursor.execute(
                        "SELECT * FROM chats WHERE id=%s AND user_id=%s",
                        (chat_id, user_id),
                    )
                    row = await cursor.fetchone()
                    if row:
                        return self._chat_from_row(row)
                selected_id = chat_id or str(uuid4())
                now = utc_now()
                await cursor.execute(
                    """
                    INSERT INTO chats(
                        id, user_id, summary, estimated_tokens,
                        codex_thread_id, created_at, updated_at
                    ) VALUES (%s,%s,'',0,NULL,%s,%s)
                    """,
                    (selected_id, user_id, now, now),
                )
            await connection.commit()
        return ChatSession(
            id=selected_id,
            user_id=user_id,
            created_at=now,
            updated_at=now,
        )

    async def get_chat(
        self, chat_id: str, user_id: str | None = None
    ) -> ChatSession | None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                if user_id:
                    await cursor.execute(
                        "SELECT * FROM chats WHERE id=%s AND user_id=%s",
                        (chat_id, user_id),
                    )
                else:
                    await cursor.execute("SELECT * FROM chats WHERE id=%s", (chat_id,))
                row = await cursor.fetchone()
        return self._chat_from_row(row) if row else None

    async def append_message(self, message: ChatMessage) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            try:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
                        INSERT INTO messages(
                            id, chat_id, role, content, metadata_json, created_at
                        ) VALUES (%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            message.id,
                            message.chat_id,
                            message.role,
                            message.content,
                            json.dumps(message.metadata, ensure_ascii=False),
                            message.created_at,
                        ),
                    )
                    await cursor.execute(
                        "UPDATE chats SET updated_at=%s WHERE id=%s",
                        (utc_now(), message.chat_id),
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def list_messages(self, chat_id: str) -> list[ChatMessage]:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.cursor(self._aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT * FROM messages WHERE chat_id=%s ORDER BY created_at,id",
                    (chat_id,),
                )
                rows = await cursor.fetchall()
        return [self._message_from_row(row) for row in rows]

    async def replace_compacted_history(
        self,
        chat_id: str,
        summary: str,
        keep_message_ids: list[str],
        estimated_tokens: int,
    ) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            try:
                async with connection.cursor() as cursor:
                    if keep_message_ids:
                        placeholders = ",".join("%s" for _ in keep_message_ids)
                        await cursor.execute(
                            f"DELETE FROM messages WHERE chat_id=%s "
                            f"AND id NOT IN ({placeholders})",
                            (chat_id, *keep_message_ids),
                        )
                    else:
                        await cursor.execute(
                            "DELETE FROM messages WHERE chat_id=%s", (chat_id,)
                        )
                    await cursor.execute(
                        """
                        UPDATE chats
                        SET summary=%s, estimated_tokens=%s, updated_at=%s
                        WHERE id=%s
                        """,
                        (summary, max(0, int(estimated_tokens)), utc_now(), chat_id),
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def set_codex_thread(self, chat_id: str, thread_id: str | None) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "UPDATE chats SET codex_thread_id=%s, updated_at=%s WHERE id=%s",
                    (thread_id, utc_now(), chat_id),
                )
            await connection.commit()

    @staticmethod
    def _metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(str(value or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _chat_from_row(row: dict[str, Any]) -> ChatSession:
        return ChatSession(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            summary=str(row.get("summary") or ""),
            estimated_tokens=int(row.get("estimated_tokens") or 0),
            codex_thread_id=(
                str(row["codex_thread_id"])
                if row.get("codex_thread_id") is not None
                else None
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @classmethod
    def _message_from_row(cls, row: dict[str, Any]) -> ChatMessage:
        return ChatMessage(
            id=str(row["id"]),
            chat_id=str(row["chat_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            metadata=cls._metadata(row.get("metadata_json")),
            created_at=str(row["created_at"]),
        )
