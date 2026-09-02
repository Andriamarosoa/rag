from pathlib import Path

import pytest

from app.context.manager import ContextManager
from app.sessions.models import ChatMessage
from app.sessions.store import SessionStore


class FakeSummarizer:
    async def summarize_context(self, previous_summary, messages):
        return f"summary:{previous_summary}|" + ";".join(m.content for m in messages)


@pytest.mark.asyncio
async def test_context_compacts_old_messages(tmp_path: Path):
    store = SessionStore(tmp_path / "rag.db")
    await store.initialize()
    chat = await store.get_or_create_chat("u1")
    for i in range(8):
        await store.append_message(ChatMessage(chat_id=chat.id, role="user", content=(f"message-{i}-" + "x" * 80)))

    manager = ContextManager(
        store=store,
        summarizer=FakeSummarizer(),
        compact_at_tokens=80,
        keep_recent_tokens=45,
        summary_target_tokens=200,
    )
    snapshot = await manager.compact_if_needed(chat.id, "u1")
    assert snapshot.chat.summary.startswith("summary:")
    assert 1 <= len(snapshot.messages) < 8
