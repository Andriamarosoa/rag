from __future__ import annotations

from app.agents.registry import AgentRegistry
from app.orchestrator import Orchestrator
from app.sessions.models import ChatSession


class FakeStore:
    def __init__(self):
        self.messages = []

    async def get_or_create_chat(self, user_id: str, chat_id: str | None = None):
        return ChatSession(id=chat_id or "chat-1", user_id=user_id)

    async def get_chat(self, chat_id: str, user_id: str | None = None):
        if chat_id != "chat-1" or (user_id is not None and user_id != "user-1"):
            return None
        return ChatSession(id=chat_id, user_id="user-1")

    async def append_message(self, message):
        self.messages.append(message)

    async def list_messages(self, _chat_id):
        return list(self.messages)


class FakeGroundedAnswer:
    allowed_host = "skb.uniconsults.mu"
    UNAVAILABLE_ANSWER = "source unavailable"

    def __init__(self):
        self.calls = []

    async def rewrite_question(self, question, history, *, module=None):
        self.calls.append(("rewrite", question, module, len(history)))
        return question

    async def retrieve(self, question: str, *, module: str | None = None):
        self.calls.append(("retrieve", question, module))
        return [object()]

    async def answer_from_chunks(self, question: str, chunks, *, module=None):
        self.calls.append(("answer", question, module, len(chunks)))
        return {
            "status": "answered",
            "answer": "Réponse SKB",
            "sources": [
                {
                    "id": "c1",
                    "page_id": "spay:faq:faq",
                    "title": "FAQ",
                    "url": "http://skb.uniconsults.mu/doku.php?id=spay%3Afaq%3Afaq",
                    "module": "Payroll",
                    "section": "Login",
                }
            ],
            "citations": ["c1"],
            "module": module,
            "grounded": True,
            "actions": [],
            "matched_rules": [],
            "matched_rule": None,
        }


async def test_grounded_path_bypasses_rules_and_persists_sources():
    store = FakeStore()
    grounded = FakeGroundedAnswer()
    orchestrator = Orchestrator(
        store=store,  # type: ignore[arg-type]
        context=None,  # type: ignore[arg-type]
        rules=None,  # type: ignore[arg-type]
        agents=AgentRegistry(),
        codex=None,  # type: ignore[arg-type]
        grounded_answer=grounded,  # type: ignore[arg-type]
    )
    events = []

    async def emit(event_type, data):
        events.append((event_type, data))

    result = await orchestrator.handle_message(
        "user-1",
        None,
        "Mot de passe oublié",
        emit=emit,
        module="spay",
    )

    assert result["answer"] == "Réponse SKB"
    assert result["chat_id"] == "chat-1"
    assert grounded.calls == [
        ("rewrite", "Mot de passe oublié", "spay", 0),
        ("retrieve", "Mot de passe oublié", "spay"),
        ("answer", "Mot de passe oublié", "spay", 1),
    ]
    assert [message.role for message in store.messages] == ["user", "assistant"]
    assert store.messages[0].metadata["module"] == "spay"
    assert store.messages[1].metadata["sources"][0]["page_id"] == "spay:faq:faq"
    assert "rules.pre.started" not in [event_type for event_type, _ in events]
    assert "rag.retrieval.completed" in [event_type for event_type, _ in events]


async def test_module_continuation_reuses_question_without_duplicate_user_message():
    store = FakeStore()
    store.messages.append(
        type(
            "Message",
            (),
            {"role": "user", "content": "How do I reset my password?"},
        )()
    )
    grounded = FakeGroundedAnswer()
    orchestrator = Orchestrator(
        store=store,  # type: ignore[arg-type]
        context=None,  # type: ignore[arg-type]
        rules=None,  # type: ignore[arg-type]
        agents=AgentRegistry(),
        codex=None,  # type: ignore[arg-type]
        grounded_answer=grounded,  # type: ignore[arg-type]
    )
    events = []

    async def emit(event_type, data):
        events.append((event_type, data))

    result = await orchestrator.handle_message(
        "user-1",
        "chat-1",
        "How do I reset my password?",
        emit=emit,
        module="spay",
        continuation=True,
    )

    assert result["chat_id"] == "chat-1"
    assert [message.role for message in store.messages] == ["user", "assistant"]
    assert "message.user.reused" in [event_type for event_type, _ in events]
    assert "message.user.persisted" not in [event_type for event_type, _ in events]
