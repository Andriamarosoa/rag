from __future__ import annotations

from typing import Any

from app.agents.registry import AgentRegistry
from app.codex.service import CodexService
from app.context.manager import ContextManager
from app.flow.events import FlowEmitter, emit_flow
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

    async def handle_message(
        self,
        user_id: str,
        chat_id: str | None,
        text: str,
        emit: FlowEmitter | None = None,
    ) -> dict[str, Any]:
        await emit_flow(emit, "flow.started", user_id=user_id, requested_chat_id=chat_id)

        chat = await self.store.get_or_create_chat(user_id=user_id, chat_id=chat_id)
        await emit_flow(
            emit,
            "session.ready",
            user_id=user_id,
            chat_id=chat.id,
            existing=bool(chat_id),
            has_codex_thread=bool(chat.codex_thread_id),
        )

        await self.store.append_message(ChatMessage(chat_id=chat.id, role="user", content=text))
        await emit_flow(
            emit,
            "message.user.persisted",
            chat_id=chat.id,
            chars=len(text),
        )

        snapshot = await self.context.compact_if_needed(chat.id, user_id, emit=emit)
        rendered_context = snapshot.render()
        await emit_flow(
            emit,
            "context.ready",
            chat_id=chat.id,
            estimated_tokens=snapshot.estimated_tokens,
            rendered_chars=len(rendered_context),
        )

        pre_rule = await self.rules.match_pre(text, rendered_context, emit=emit)
        if pre_rule and pre_rule.then.get("type") == "respond":
            canonical = str(pre_rule.then.get("canonical_answer", ""))
            reformulate = bool(pre_rule.then.get("reformulate", False))
            await emit_flow(
                emit,
                "rule.action.started",
                rule_id=pre_rule.id,
                action_type="respond",
                reformulate=reformulate,
            )
            if reformulate:
                answer = await self.codex.reformulate(canonical, text, rendered_context, emit=emit)
            else:
                answer = canonical
            result: dict[str, Any] = {
                "status": "answered",
                "answer": answer,
                "matched_rule": pre_rule.id,
                "actions": [],
            }
            await emit_flow(
                emit,
                "rule.action.completed",
                rule_id=pre_rule.id,
                action_type="respond",
                status="answered",
            )
        else:
            if pre_rule:
                await emit_flow(
                    emit,
                    "rule.action.unsupported",
                    rule_id=pre_rule.id,
                    action_type=pre_rule.then.get("type"),
                )

            refreshed_chat = await self.store.get_chat(chat.id, user_id)
            await emit_flow(
                emit,
                "reasoning.started",
                available_agent_count=len(self.agents.specs()),
            )
            result = await self.codex.answer(
                user_message=text,
                rendered_context=rendered_context,
                agents=self.agents.specs(),
                thread_id=refreshed_chat.codex_thread_id if refreshed_chat else None,
                emit=emit,
            )
            await emit_flow(
                emit,
                "reasoning.completed",
                status=result.get("status"),
                suggested_agent=result.get("suggested_agent"),
            )

            if result.get("thread_id") and result.get("thread_id") != chat.codex_thread_id:
                await self.store.set_codex_thread(chat.id, result["thread_id"])
                await emit_flow(
                    emit,
                    "session.codex_thread.updated",
                    chat_id=chat.id,
                    thread_id=result["thread_id"],
                )
            result.setdefault("actions", [])

            suggested = result.get("suggested_agent")
            if suggested and self.agents.get(suggested):
                spec = self.agents.get(suggested).spec
                action = {
                    "type": "suggest_agent",
                    "agent": suggested,
                    "label": suggested.replace("_", " ").title(),
                    "arguments": result.get("suggested_agent_args") or {},
                    "requires_confirmation": spec.requires_confirmation,
                }
                result["actions"].append(action)
                await emit_flow(
                    emit,
                    "agent.suggested",
                    agent=suggested,
                    source="model",
                    requires_confirmation=spec.requires_confirmation,
                    arguments=action["arguments"],
                )

        await emit_flow(
            emit,
            "rules.post.started",
            result_status=result.get("status"),
        )
        post_rules = self.rules.match_post(result)
        if not post_rules:
            await emit_flow(emit, "rules.post.no_match", result_status=result.get("status"))

        for rule in post_rules:
            await emit_flow(
                emit,
                "rules.post.matched",
                rule_id=rule.id,
                action_type=rule.then.get("type"),
            )
            then = rule.then
            if then.get("type") == "suggest_agent" and self.agents.get(str(then.get("agent"))):
                action = {
                    "type": "suggest_agent",
                    "agent": then["agent"],
                    "label": then.get("label", then["agent"]),
                    "arguments": {},
                    "requires_confirmation": bool(then.get("requires_confirmation", True)),
                    "rule_id": rule.id,
                }
                result["actions"].append(action)
                await emit_flow(
                    emit,
                    "agent.suggested",
                    agent=then["agent"],
                    source="post_rule",
                    rule_id=rule.id,
                    requires_confirmation=action["requires_confirmation"],
                    arguments={},
                )

        answer = result.get("answer")
        if not answer and result.get("status") in {"not_found", "insufficient_information"}:
            answer = "I do not have enough reliable information to answer this question."
            result["answer"] = answer
            await emit_flow(
                emit,
                "response.fallback.applied",
                reason=result.get("status"),
            )

        await self.store.append_message(
            ChatMessage(
                chat_id=chat.id,
                role="assistant",
                content=str(answer or ""),
                metadata={"status": result.get("status"), "matched_rule": result.get("matched_rule")},
            )
        )
        await emit_flow(
            emit,
            "message.assistant.persisted",
            chat_id=chat.id,
            status=result.get("status"),
            chars=len(str(answer or "")),
        )

        await self.context.compact_if_needed(chat.id, user_id, emit=emit)

        result["chat_id"] = chat.id
        await emit_flow(
            emit,
            "flow.completed",
            chat_id=chat.id,
            status=result.get("status"),
            action_count=len(result.get("actions") or []),
            matched_rule=result.get("matched_rule"),
        )
        return result
