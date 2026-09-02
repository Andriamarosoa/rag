from __future__ import annotations

from time import perf_counter
from typing import Any

from app.agents.registry import AgentRegistry
from app.codex.service import CodexService
from app.context.manager import ContextManager
from app.flow.events import FlowEmitter, emit_flow
from app.rules.engine import RuleEngine
from app.rules.models import FunctionalRule
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

    async def _apply_rule_actions(
        self,
        rule: FunctionalRule,
        result: dict[str, Any],
        emit: FlowEmitter | None,
        *,
        source: str,
        allow_model_reformulation: bool,
    ) -> None:
        """Execute a rule's ordered `then` action list locally."""
        result.setdefault("actions", [])
        action_count = len(rule.then)

        for action_index, action in enumerate(rule.then):
            action_type = str(action.get("type") or "").strip()
            await emit_flow(
                emit,
                "rule.action.started",
                rule_id=rule.id,
                action_type=action_type or None,
                action_index=action_index,
                action_count=action_count,
                source=source,
            )

            if action_type == "respond":
                canonical = str(action.get("canonical_answer", ""))
                reformulate = bool(action.get("reformulate", False))
                if reformulate and allow_model_reformulation and result.get("answer"):
                    answer = str(result["answer"]).strip()
                else:
                    answer = canonical

                result.update(
                    {
                        "status": "answered",
                        "answer": answer,
                        "matched_rule": rule.id,
                    }
                )
                await emit_flow(
                    emit,
                    "rule.action.completed",
                    rule_id=rule.id,
                    action_type="respond",
                    action_index=action_index,
                    action_count=action_count,
                    status="answered",
                    reformulate=reformulate,
                    second_model_call=False,
                )
                continue

            if action_type == "suggest_agent":
                agent_name = str(action.get("agent") or "").strip()
                agent = self.agents.get(agent_name) if agent_name else None
                if agent is None:
                    await emit_flow(
                        emit,
                        "rule.action.unsupported",
                        rule_id=rule.id,
                        action_type=action_type,
                        action_index=action_index,
                        action_count=action_count,
                        reason="unknown_agent",
                        agent=agent_name or None,
                    )
                    continue

                arguments = action.get("arguments")
                if not isinstance(arguments, dict):
                    arguments = {}
                requires_confirmation = bool(
                    action.get("requires_confirmation", agent.spec.requires_confirmation)
                )
                ui_action = {
                    "type": "suggest_agent",
                    "agent": agent_name,
                    "label": action.get("label", agent_name.replace("_", " ").title()),
                    "arguments": arguments,
                    "requires_confirmation": requires_confirmation,
                    "rule_id": rule.id,
                }
                result["actions"].append(ui_action)
                await emit_flow(
                    emit,
                    "agent.suggested",
                    agent=agent_name,
                    source=source,
                    rule_id=rule.id,
                    action_index=action_index,
                    requires_confirmation=requires_confirmation,
                    arguments=arguments,
                )
                await emit_flow(
                    emit,
                    "rule.action.completed",
                    rule_id=rule.id,
                    action_type="suggest_agent",
                    action_index=action_index,
                    action_count=action_count,
                    status="suggested",
                )
                continue

            await emit_flow(
                emit,
                "rule.action.unsupported",
                rule_id=rule.id,
                action_type=action_type or None,
                action_index=action_index,
                action_count=action_count,
                reason="unsupported_action_type",
            )

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

        semantic_rules = self.rules.semantic_pre_rules()
        rule_decision_started_at = perf_counter()
        await emit_flow(
            emit,
            "rules.pre.started",
            candidate_count=len(semantic_rules),
            integrated_with_reasoning=True,
            timing_scope="integrated_assistant_decision",
        )

        refreshed_chat = await self.store.get_chat(chat.id, user_id)
        await emit_flow(
            emit,
            "reasoning.started",
            available_agent_count=len(self.agents.specs()),
            semantic_rule_count=len(semantic_rules),
            integrated_rule_matching=True,
        )

        result = await self.codex.answer_with_rules(
            user_message=text,
            rendered_context=rendered_context,
            rules=[rule.model_dump() for rule in semantic_rules],
            agents=self.agents.specs(),
            thread_id=refreshed_chat.codex_thread_id if refreshed_chat else None,
            emit=emit,
        )

        rule_decision_elapsed_ms = round((perf_counter() - rule_decision_started_at) * 1000, 1)
        proposed_rule_id = result.get("matched_rule")
        pre_rule = await self.rules.resolve_pre_decision(
            result,
            emit=emit,
            decision_elapsed_ms=rule_decision_elapsed_ms,
        )

        if proposed_rule_id and pre_rule is None:
            result.update(
                {
                    "matched_rule": None,
                    "status": "insufficient_information",
                    "answer": None,
                    "suggested_agent": None,
                    "suggested_agent_args": {},
                    "actions": [],
                }
            )
            await emit_flow(
                emit,
                "reasoning.rule_answer.discarded",
                proposed_rule_id=proposed_rule_id,
                reason="rule_not_accepted",
            )

        if result.get("thread_id") and result.get("thread_id") != chat.codex_thread_id:
            await self.store.set_codex_thread(chat.id, result["thread_id"])
            await emit_flow(
                emit,
                "session.codex_thread.updated",
                chat_id=chat.id,
                thread_id=result["thread_id"],
            )

        if pre_rule:
            # A matched functional rule is authoritative. Model-proposed actions are discarded;
            # only the ordered actions configured in `then` are applied.
            result["actions"] = []
            result["suggested_agent"] = None
            result["suggested_agent_args"] = {}
            result["matched_rule"] = pre_rule.id
            await self._apply_rule_actions(
                pre_rule,
                result,
                emit,
                source="pre_rule",
                allow_model_reformulation=True,
            )
        else:
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
            "reasoning.completed",
            status=result.get("status"),
            matched_rule=result.get("matched_rule"),
            suggested_agent=result.get("suggested_agent"),
            integrated_rule_matching=True,
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
            action_types = [str(action.get("type") or "") for action in rule.then]
            await emit_flow(
                emit,
                "rules.post.matched",
                rule_id=rule.id,
                action_type=action_types[0] if action_types else None,
                action_types=action_types,
                action_count=len(rule.then),
            )
            await self._apply_rule_actions(
                rule,
                result,
                emit,
                source="post_rule",
                allow_model_reformulation=False,
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
