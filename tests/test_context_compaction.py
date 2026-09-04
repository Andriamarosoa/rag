import pytest

from app.context.manager import ContextManager
from app.sessions.models import ChatMessage, ChatSession


class FakeSummarizer:
    async def summarize_context(self, previous_summary, messages, emit=None):
        return f"summary:{previous_summary}|" + ";".join(m.content for m in messages)


class FakeStore:
    def __init__(self):
        self.chat = ChatSession(id="chat-1", user_id="u1")
        self.messages = []

    async def get_or_create_chat(self, user_id, chat_id=None):
        return self.chat

    async def get_chat(self, chat_id, user_id=None):
        if chat_id != self.chat.id or (user_id and user_id != self.chat.user_id):
            return None
        return self.chat

    async def append_message(self, message):
        self.messages.append(message)

    async def list_messages(self, chat_id):
        return list(self.messages)

    async def replace_compacted_history(
        self, chat_id, summary, keep_message_ids, estimated_tokens
    ):
        keep = set(keep_message_ids)
        self.messages = [message for message in self.messages if message.id in keep]
        self.chat.summary = summary
        self.chat.estimated_tokens = estimated_tokens


@pytest.mark.asyncio
async def test_context_compacts_old_messages():
    store = FakeStore()
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
    events = []

    async def emit(event_type, data):
        events.append((event_type, data))

    snapshot = await manager.compact_if_needed(chat.id, "u1", emit=emit)
    assert snapshot.chat.summary.startswith("summary:")
    assert 1 <= len(snapshot.messages) < 8
    event_types = [event_type for event_type, _ in events]
    assert "context.snapshot" in event_types
    assert "context.compaction.started" in event_types
    assert "context.compaction.completed" in event_types
