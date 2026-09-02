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
    MAX_RULE_REFERENCE_DEPTH = 16
    MAX_RULE_CONTROL_DEPTH = 32
    _CONTROL_KEYS = {"ref", "then", "catch"}

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

    @staticmethod
    def _branch_items(branch: Any) -> list[Any]:
        if branch is None:
            return []
        if isinstance(branch, list):
            return branch
        return [branch]

    @classmethod
    def _reference_overrides(
        cls,
        node: dict[str, Any],
        inherited: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        overrides = dict(inherited or {})
        for key, value in node.items():
            if key not in cls._CONTROL_KEYS:
                overrides[key] = value
        return overrides

    async def _execute_reference(
        self,
        rule_id: str,
        result: dict[str, Any],
        emit: FlowEmitter | None,
        *,
        source: str,
        allow_model_reformulation: bool,
        origin_rule_id: str,
        reference_stack: tuple[str, ...],
        control_depth: int,
        overrides: dict[str, Any] | None = None,
        action_index: int | None = None,
        action_count: int | None = None,
    ) -> bool:
        referenced_rule_id = rule_id.strip()
        await emit_flow(
            emit,
            "rule.reference.started",
            rule_id=reference_stack[-1] if reference_stack else origin_rule_id,
            origin_rule_id=origin_rule_id,
            referenced_rule_id=referenced_rule_id or None,
            action_index=action_index,
            action_count=action_count,
            reference_depth=len(reference_stack),
            control_depth=control_depth,
            overrides=overrides or {},
            source=source,
        )

        if not referenced_rule_id:
            await emit_flow(
                emit,
                "rule.reference.rejected",
                origin_rule_id=origin_rule_id,
                referenced_rule_id=None,
                action_index=action_index,
                reason="empty_rule_reference",
            )
            return False

        if referenced_rule_id in reference_stack:
            await emit_flow(
                emit,
                "rule.reference.rejected",
                origin_rule_id=origin_rule_id,
                referenced_rule_id=referenced_rule_id,
                action_index=action_index,
                reason="cyclic_rule_reference",
                reference_path=[*reference_stack, referenced_rule_id],
            )
            return False

        if len(reference_stack) >= self.MAX_RULE_REFERENCE_DEPTH:
            await emit_flow(
                emit,
                "rule.reference.rejected",
                origin_rule_id=origin_rule_id,
                referenced_rule_id=referenced_rule_id,
                action_index=action_index,
                reason="max_rule_reference_depth",
                max_depth=self.MAX_RULE_REFERENCE_DEPTH,
            )
            return False

        referenced_rule = self.rules.get_rule(referenced_rule_id)
        if referenced_rule is None:
            await emit_flow(
                emit,
                "rule.reference.rejected",
                origin_rule_id=origin_rule_id,
                referenced_rule_id=referenced_rule_id,
                action_index=action_index,
                reason="unknown_or_disabled_rule_reference",
            )
            return False

        success = await self._execute_branch(
            referenced_rule.then,
            referenced_rule,
            result,
            emit,
            source="rule_reference",
            allow_model_reformulation=allow_model_reformulation,
            origin_rule_id=origin_rule_id,
            reference_stack=(*reference_stack, referenced_rule_id),
            control_depth=control_depth + 1,
            inherited_overrides=overrides,
        )

        await emit_flow(
            emit,
            "rule.reference.completed" if success else "rule.reference.failed",
            rule_id=referenced_rule_id,
            origin_rule_id=origin_rule_id,
            referenced_rule_id=referenced_rule_id,
            action_index=action_index,
            action_count=action_count,
            overrides=overrides or {},
        )
        return success

    async def _execute_concrete_action(
        self,
        action: dict[str, Any],
        rule: FunctionalRule,
        result: dict[str, Any],
        emit: FlowEmitter | None,
        *,
        source: str,
        origin_rule_id: str,
        action_index: int,
        action_count: int,
        control_depth: int,
        allow_model_reformulation: bool,
    ) -> bool:
        action_type = str(action.get("type") or "").strip()
        await emit_flow(
            emit,
            "rule.action.started",
            rule_id=rule.id,
            origin_rule_id=origin_rule_id,
            action_type=action_type or None,
            action_index=action_index,
            action_count=action_count,
            control_depth=control_depth,
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
                    "matched_rule": origin_rule_id,
                }
            )
            await emit_flow(
                emit,
                "rule.action.completed",
                rule_id=rule.id,
                origin_rule_id=origin_rule_id,
                action_type="respond",
                action_index=action_index,
                action_count=action_count,
                status="answered",
                reformulate=reformulate,
                second_model_call=False,
            )
            return True

        if action_type == "suggest_agent":
            agent_name = str(action.get("agent") or "").strip()
            agent = self.agents.get(agent_name) if agent_name else None
            if agent is None:
                await emit_flow(
                    emit,
                    "rule.action.failed",
                    rule_id=rule.id,
                    origin_rule_id=origin_rule_id,
                    action_type=action_type,
                    action_index=action_index,
                    action_count=action_count,
                    reason="unknown_agent",
                    agent=agent_name or None,
                )
                return False

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
                "origin_rule_id": origin_rule_id,
            }
            result["actions"].append(ui_action)
            await emit_flow(
                emit,
                "agent.suggested",
                agent=agent_name,
                source=source,
                rule_id=rule.id,
                origin_rule_id=origin_rule_id,
                action_index=action_index,
                requires_confirmation=requires_confirmation,
                arguments=arguments,
            )
            await emit_flow(
                emit,
                "rule.action.completed",
                rule_id=rule.id,
                origin_rule_id=origin_rule_id,
                action_type="suggest_agent",
                action_index=action_index,
                action_count=action_count,
                status="suggested",
            )
            return True

        await emit_flow(
            emit,
            "rule.action.failed",
            rule_id=rule.id,
            origin_rule_id=origin_rule_id,
            action_type=action_type or None,
            action_index=action_index,
            action_count=action_count,
            reason="unsupported_action_type",
        )
        return False

    async def _execute_catch(
        self,
        catch_branch: Any,
        rule: FunctionalRule,
        result: dict[str, Any],
        emit: FlowEmitter | None,
        *,
        source: str,
        allow_model_reformulation: bool,
        origin_rule_id: str,
        reference_stack: tuple[str, ...],
        control_depth: int,
    ) -> bool:
        await emit_flow(
            emit,
            "rule.catch.started",
            rule_id=rule.id,
            origin_rule_id=origin_rule_id,
            control_depth=control_depth,
            source=source,
        )
        success = await self._execute_branch(
            catch_branch,
            rule,
            result,
            emit,
            source="catch",
            allow_model_reformulation=allow_model_reformulation,
            origin_rule_id=origin_rule_id,
            reference_stack=reference_stack,
            control_depth=control_depth + 1,
        )
        await emit_flow(
            emit,
            "rule.catch.completed" if success else "rule.catch.failed",
            rule_id=rule.id,
            origin_rule_id=origin_rule_id,
            control_depth=control_depth,
        )
        return success

    async def _execute_rule_node(
        self,
        item: Any,
        rule: FunctionalRule,
        result: dict[str, Any],
        emit: FlowEmitter | None,
        *,
        source: str,
        allow_model_reformulation: bool,
        origin_rule_id: str,
        reference_stack: tuple[str, ...],
        control_depth: int,
        inherited_overrides: dict[str, Any] | None,
        action_index: int,
        action_count: int,
    ) -> bool:
        if control_depth > self.MAX_RULE_CONTROL_DEPTH:
            await emit_flow(
                emit,
                "rule.control.rejected",
                rule_id=rule.id,
                origin_rule_id=origin_rule_id,
                reason="max_rule_control_depth",
                max_depth=self.MAX_RULE_CONTROL_DEPTH,
            )
            return False

        if isinstance(item, str):
            return await self._execute_reference(
                item,
                result,
                emit,
                source=source,
                allow_model_reformulation=allow_model_reformulation,
                origin_rule_id=origin_rule_id,
                reference_stack=reference_stack,
                control_depth=control_depth,
                overrides=inherited_overrides,
                action_index=action_index,
                action_count=action_count,
            )

        if not isinstance(item, dict):
            await emit_flow(
                emit,
                "rule.control.rejected",
                rule_id=rule.id,
                origin_rule_id=origin_rule_id,
                action_index=action_index,
                reason="invalid_rule_node",
                node_type=type(item).__name__,
            )
            return False

        nested_then = item.get("then")
        nested_catch = item.get("catch")

        if "ref" in item:
            ref_id = str(item.get("ref") or "")
            overrides = self._reference_overrides(item, inherited_overrides)
            success = await self._execute_reference(
                ref_id,
                result,
                emit,
                source=source,
                allow_model_reformulation=allow_model_reformulation,
                origin_rule_id=origin_rule_id,
                reference_stack=reference_stack,
                control_depth=control_depth,
                overrides=overrides,
                action_index=action_index,
                action_count=action_count,
            )
        else:
            action_type = str(item.get("type") or "").strip()
            if action_type:
                action = {
                    key: value
                    for key, value in item.items()
                    if key not in {"then", "catch"}
                }
                if inherited_overrides:
                    action.update(inherited_overrides)
                success = await self._execute_concrete_action(
                    action,
                    rule,
                    result,
                    emit,
                    source=source,
                    origin_rule_id=origin_rule_id,
                    action_index=action_index,
                    action_count=action_count,
                    control_depth=control_depth,
                    allow_model_reformulation=allow_model_reformulation,
                )
            elif "then" in item:
                await emit_flow(
                    emit,
                    "rule.then.started",
                    rule_id=rule.id,
                    origin_rule_id=origin_rule_id,
                    control_depth=control_depth,
                    source=source,
                )
                success = await self._execute_branch(
                    nested_then,
                    rule,
                    result,
                    emit,
                    source="then",
                    allow_model_reformulation=allow_model_reformulation,
                    origin_rule_id=origin_rule_id,
                    reference_stack=reference_stack,
                    control_depth=control_depth + 1,
                    inherited_overrides=inherited_overrides,
                )
                await emit_flow(
                    emit,
                    "rule.then.completed" if success else "rule.then.failed",
                    rule_id=rule.id,
                    origin_rule_id=origin_rule_id,
                    control_depth=control_depth,
                )
                nested_then = None
            else:
                await emit_flow(
                    emit,
                    "rule.control.rejected",
                    rule_id=rule.id,
                    origin_rule_id=origin_rule_id,
                    action_index=action_index,
                    reason="rule_node_has_no_type_ref_or_then",
                )
                success = False

        if success and nested_then is not None:
            await emit_flow(
                emit,
                "rule.then.started",
                rule_id=rule.id,
                origin_rule_id=origin_rule_id,
                control_depth=control_depth,
                source=source,
            )
            success = await self._execute_branch(
                nested_then,
                rule,
                result,
                emit,
                source="then",
                allow_model_reformulation=allow_model_reformulation,
                origin_rule_id=origin_rule_id,
                reference_stack=reference_stack,
                control_depth=control_depth + 1,
            )
            await emit_flow(
                emit,
                "rule.then.completed" if success else "rule.then.failed",
                rule_id=rule.id,
                origin_rule_id=origin_rule_id,
                control_depth=control_depth,
            )

        if not success and nested_catch is not None:
            return await self._execute_catch(
                nested_catch,
                rule,
                result,
                emit,
                source=source,
                allow_model_reformulation=allow_model_reformulation,
                origin_rule_id=origin_rule_id,
                reference_stack=reference_stack,
                control_depth=control_depth,
            )

        return success

    async def _execute_branch(
        self,
        branch: Any,
        rule: FunctionalRule,
        result: dict[str, Any],
        emit: FlowEmitter | None,
        *,
        source: str,
        allow_model_reformulation: bool,
        origin_rule_id: str,
        reference_stack: tuple[str, ...],
        control_depth: int,
        inherited_overrides: dict[str, Any] | None = None,
    ) -> bool:
        items = self._branch_items(branch)
        for action_index, item in enumerate(items):
            success = await self._execute_rule_node(
                item,
                rule,
                result,
                emit,
                source=source,
                allow_model_reformulation=allow_model_reformulation,
                origin_rule_id=origin_rule_id,
                reference_stack=reference_stack,
                control_depth=control_depth,
                inherited_overrides=inherited_overrides,
                action_index=action_index,
                action_count=len(items),
            )
            if not success:
                return False
        return True

    async def _apply_rule_actions(
        self,
        rule: FunctionalRule,
        result: dict[str, Any],
        emit: FlowEmitter | None,
        *,
        source: str,
        allow_model_reformulation: bool,
        origin_rule_id: str | None = None,
    ) -> bool:
        """Execute a rule's recursive `then`/`catch` DSL locally."""
        result.setdefault("actions", [])
        origin_rule_id = origin_rule_id or rule.id
        return await self._execute_branch(
            rule.then,
            rule,
            result,
            emit,
            source=source,
            allow_model_reformulation=allow_model_reformulation,
            origin_rule_id=origin_rule_id,
            reference_stack=(rule.id,),
            control_depth=0,
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
                origin_rule_id=pre_rule.id,
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
            action_types = self.rules.action_labels(rule)
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
                origin_rule_id=rule.id,
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
