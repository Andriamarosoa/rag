from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.sessions.models import ChatMessage, ChatSession
from app.sessions.store import SessionStore


class Summarizer(Protocol):
    async def summarize_context(self, previous_summary: str, messages: list[ChatMessage]) -> str: ...


@dataclass(slots=True)
class ContextSnapshot:
    chat: ChatSession
    messages: list[ChatMessage]
    estimated_tokens: int

    def render(self) -> str:
        chunks: list[str] = []
        if self.chat.summary:
            chunks.append("[ROLLING CONTEXT SUMMARY]\n" + self.chat.summary)
        if self.messages:
            transcript = "\n".join(f"{m.role.upper()}: {m.content}" for m in self.messages)
            chunks.append("[RECENT MESSAGES]\n" + transcript)
        return "\n\n".join(chunks)


class ContextManager:
    """Rolling context manager inspired by agent harness compaction.

    Once the estimated context crosses `compact_at_tokens`, old messages are summarized
    and only the most recent `keep_recent_tokens` remain verbatim.
    """

    def __init__(
        self,
        store: SessionStore,
        summarizer: Summarizer,
        compact_at_tokens: int = 50_000,
        keep_recent_tokens: int = 12_000,
        summary_target_tokens: int = 8_000,
    ):
        self.store = store
        self.summarizer = summarizer
        self.compact_at_tokens = compact_at_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.summary_target_tokens = summary_target_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Dependency-free approximation. Replace with the active model tokenizer if desired.
        return max(1, len(text) // 4)

    async def snapshot(self, chat_id: str, user_id: str) -> ContextSnapshot:
        chat = await self.store.get_chat(chat_id, user_id)
        if not chat:
            raise ValueError("chat_not_found")
        messages = await self.store.list_messages(chat_id)
        total = self.estimate_tokens(chat.summary) + sum(self.estimate_tokens(m.content) for m in messages)
        return ContextSnapshot(chat=chat, messages=messages, estimated_tokens=total)

    async def compact_if_needed(self, chat_id: str, user_id: str) -> ContextSnapshot:
        snapshot = await self.snapshot(chat_id, user_id)
        if snapshot.estimated_tokens < self.compact_at_tokens:
            return snapshot

        recent: list[ChatMessage] = []
        recent_tokens = 0
        for message in reversed(snapshot.messages):
            cost = self.estimate_tokens(message.content)
            if recent and recent_tokens + cost > self.keep_recent_tokens:
                break
            recent.append(message)
            recent_tokens += cost
        recent.reverse()

        recent_ids = {m.id for m in recent}
        archived = [m for m in snapshot.messages if m.id not in recent_ids]
        if not archived:
            return snapshot

        summary = await self.summarizer.summarize_context(snapshot.chat.summary, archived)
        # Hard safety bound. This is approximate but prevents unbounded rolling summaries.
        max_chars = self.summary_target_tokens * 4
        if len(summary) > max_chars:
            summary = summary[-max_chars:]

        estimated = self.estimate_tokens(summary) + recent_tokens
        await self.store.replace_compacted_history(
            chat_id=chat_id,
            summary=summary,
            keep_message_ids=[m.id for m in recent],
            estimated_tokens=estimated,
        )
        return await self.snapshot(chat_id, user_id)
