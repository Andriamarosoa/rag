from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

from .models import ChatMessage, ChatSession, utc_now


class SessionStore:
    """SQLite persistence for users/chats/messages.

    SQLite calls are moved to worker threads so the FastAPI event loop remains free.
    """

    def __init__(self, database_path: Path):
        self.database_path = database_path

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._initialize_sync)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_sync(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    estimated_tokens INTEGER NOT NULL DEFAULT 0,
                    codex_thread_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at ASC);
                """
            )

    async def get_or_create_chat(self, user_id: str, chat_id: str | None = None) -> ChatSession:
        return await asyncio.to_thread(self._get_or_create_chat_sync, user_id, chat_id)

    def _get_or_create_chat_sync(self, user_id: str, chat_id: str | None) -> ChatSession:
        with self._connect() as conn:
            if chat_id:
                row = conn.execute("SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
                if row:
                    return self._chat_from_row(row)

            chat_id = chat_id or str(uuid4())
            now = utc_now()
            conn.execute(
                "INSERT INTO chats(id, user_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (chat_id, user_id, now, now),
            )
            conn.commit()
            return ChatSession(id=chat_id, user_id=user_id, created_at=now, updated_at=now)

    async def get_chat(self, chat_id: str, user_id: str | None = None) -> ChatSession | None:
        return await asyncio.to_thread(self._get_chat_sync, chat_id, user_id)

    def _get_chat_sync(self, chat_id: str, user_id: str | None) -> ChatSession | None:
        with self._connect() as conn:
            if user_id:
                row = conn.execute("SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
            else:
                row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
            return self._chat_from_row(row) if row else None

    async def append_message(self, message: ChatMessage) -> None:
        await asyncio.to_thread(self._append_message_sync, message)

    def _append_message_sync(self, message: ChatMessage) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(id, chat_id, role, content, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (message.id, message.chat_id, message.role, message.content, json.dumps(message.metadata), message.created_at),
            )
            conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (utc_now(), message.chat_id))
            conn.commit()

    async def list_messages(self, chat_id: str) -> list[ChatMessage]:
        return await asyncio.to_thread(self._list_messages_sync, chat_id)

    def _list_messages_sync(self, chat_id: str) -> list[ChatMessage]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC", (chat_id,)).fetchall()
        return [
            ChatMessage(
                id=row["id"],
                chat_id=row["chat_id"],
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def replace_compacted_history(
        self,
        chat_id: str,
        summary: str,
        keep_message_ids: list[str],
        estimated_tokens: int,
    ) -> None:
        await asyncio.to_thread(
            self._replace_compacted_history_sync,
            chat_id,
            summary,
            keep_message_ids,
            estimated_tokens,
        )

    def _replace_compacted_history_sync(
        self,
        chat_id: str,
        summary: str,
        keep_message_ids: list[str],
        estimated_tokens: int,
    ) -> None:
        with self._connect() as conn:
            if keep_message_ids:
                placeholders = ",".join("?" for _ in keep_message_ids)
                conn.execute(
                    f"DELETE FROM messages WHERE chat_id = ? AND id NOT IN ({placeholders})",
                    [chat_id, *keep_message_ids],
                )
            else:
                conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute(
                "UPDATE chats SET summary = ?, estimated_tokens = ?, updated_at = ? WHERE id = ?",
                (summary, estimated_tokens, utc_now(), chat_id),
            )
            conn.commit()

    async def set_codex_thread(self, chat_id: str, thread_id: str | None) -> None:
        await asyncio.to_thread(self._set_codex_thread_sync, chat_id, thread_id)

    def _set_codex_thread_sync(self, chat_id: str, thread_id: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chats SET codex_thread_id = ?, updated_at = ? WHERE id = ?",
                (thread_id, utc_now(), chat_id),
            )
            conn.commit()

    @staticmethod
    def _chat_from_row(row: sqlite3.Row) -> ChatSession:
        return ChatSession(
            id=row["id"],
            user_id=row["user_id"],
            summary=row["summary"],
            estimated_tokens=row["estimated_tokens"],
            codex_thread_id=row["codex_thread_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
