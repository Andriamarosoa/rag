from __future__ import annotations

from typing import Any

from app.agents.registry import AgentRegistry
from app.codex.service import CodexService
from app.context.manager import ContextManager
from app.rules.engine import RuleEngine
from app.sessions.models import ChatMessage
from app.sessions.store import SessionStore


class Orchestrator:
    def __init__(
        self,
        store: SessionStore,
        context: ContextManager,
        rules: RuleEngine,
        agents: AgentRegistry,
        codex: CodexService,
    ):
        self.store = store
        self.context = context
        self.rules = rules
        self.agents = agents
        self.codex = codex

    async def handle_message(self, user_id: str, chat_id: str | None, text: str) -> dict[str, Any]:
        chat = await self.store.get_or_create_chat(user_id=user_id, chat_id=chat_id)
        await self.store.append_message(ChatMessage(chat_id=chat.id, role="user", content=text))

        snapshot = await self.context.compact_if_needed(chat.id, user_id)
        rendered_context = snapshot.render()

        pre_rule = await self.rules.match_pre(text, rendered_context)
        if pre_rule and pre_rule.then.get("type") == "respond":
            canonical = str(pre_rule.then.get("canonical_answer", ""))
            if pre_rule.then.get("reformulate", False):
                answer = await self.codex.reformulate(canonical, text, rendered_context)
            else:
                answer = canonical
            result: dict[str, Any] = {
                "status": "answered",
                "answer": answer,
                "matched_rule": pre_rule.id,
                "actions": [],
            }
        else:
            refreshed_chat = await self.store.get_chat(chat.id, user_id)
            result = await self.codex.answer(
                user_message=text,
                rendered_context=rendered_context,
                agents=self.agents.specs(),
                thread_id=refreshed_chat.codex_thread_id if refreshed_chat else None,
            )
            if result.get("thread_id") and result.get("thread_id") != chat.codex_thread_id:
                await self.store.set_codex_thread(chat.id, result["thread_id"])
            result.setdefault("actions", [])

            suggested = result.get("suggested_agent")
            if suggested and self.agents.get(suggested):
                spec = self.agents.get(suggested).spec
                result["actions"].append(
                    {
                        "type": "suggest_agent",
                        "agent": suggested,
                        "label": suggested.replace("_", " ").title(),
                        "arguments": result.get("suggested_agent_args") or {},
                        "requires_confirmation": spec.requires_confirmation,
                    }
                )

        for rule in self.rules.match_post(result):
            then = rule.then
            if then.get("type") == "suggest_agent" and self.agents.get(str(then.get("agent"))):
                result["actions"].append(
                    {
                        "type": "suggest_agent",
                        "agent": then["agent"],
                        "label": then.get("label", then["agent"]),
                        "arguments": {},
                        "requires_confirmation": bool(then.get("requires_confirmation", True)),
                        "rule_id": rule.id,
                    }
                )

        answer = result.get("answer")
        if not answer and result.get("status") in {"not_found", "insufficient_information"}:
            answer = "I do not have enough reliable information to answer this question."
            result["answer"] = answer

        await self.store.append_message(
            ChatMessage(
                chat_id=chat.id,
                role="assistant",
                content=str(answer or ""),
                metadata={"status": result.get("status"), "matched_rule": result.get("matched_rule")},
            )
        )
        await self.context.compact_if_needed(chat.id, user_id)

        result["chat_id"] = chat.id
        return result
