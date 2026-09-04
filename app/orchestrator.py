from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

from app.agents.registry import AgentRegistry
from app.codex.service import CodexService
from app.context.manager import ContextManager
from app.flow.events import FlowEmitter, emit_flow
from app.rules.engine import RuleEngine
from app.rules.models import FunctionalRule
from app.sessions.models import ChatMessage
from app.sessions.store import SessionStore

if TYPE_CHECKING:
    from app.grounded_answer import GroundedAnswerService


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
        grounded_answer: GroundedAnswerService | None = None,
    ):
        self.store = store
        self.context = context
        self.rules = rules
        self.agents = agents
        self.codex = codex
        self.grounded_answer = grounded_answer

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

    @staticmethod
    def _append_rule_output(
        result: dict[str, Any],
        *,
        content: str,
        rule_id: str,
        origin_rule_id: str,
        source: str,
    ) -> None:
        text = str(content or "").strip()
        if not text:
            return
        outputs = result.setdefault("_rule_outputs", [])
        if not isinstance(outputs, list):
            outputs = []
            result["_rule_outputs"] = outputs
        outputs.append(
            {
                "rule_id": rule_id,
                "origin_rule_id": origin_rule_id,
                "source": source,
                "content": text,
            }
        )

    @staticmethod
    def _compose_rule_outputs(result: dict[str, Any]) -> list[dict[str, Any]]:
        raw_outputs = result.get("_rule_outputs")
        if not isinstance(raw_outputs, list):
            result["rule_outputs"] = []
            result.pop("_rule_outputs", None)
            return []

        outputs: list[dict[str, Any]] = []
        parts: list[str] = []
        seen_text: set[str] = set()

        for item in raw_outputs:
            if not isinstance(item, dict):
                continue
            text = str(item.get("content") or "").strip()
            if not text:
                continue
            dedupe_key = " ".join(text.split()).casefold()
            if dedupe_key in seen_text:
                continue
            seen_text.add(dedupe_key)
            normalized = {
                "rule_id": str(item.get("rule_id") or ""),
                "origin_rule_id": str(item.get("origin_rule_id") or ""),
                "source": str(item.get("source") or ""),
                "content": text,
            }
            outputs.append(normalized)
            parts.append(text)

        result["rule_outputs"] = outputs
        result.pop("_rule_outputs", None)

        if parts:
            result["answer"] = "\n\n".join(parts)
            result["status"] = "answered"

        return outputs

    @staticmethod
    def _compose_post_messages(result: dict[str, Any]) -> list[str]:
        raw_messages = result.pop("_post_messages", [])
        if not isinstance(raw_messages, list):
            raw_messages = []

        messages: list[str] = []
        seen: set[str] = set()
        for value in raw_messages:
            text = str(value or "").strip()
            if not text:
                continue
            key = " ".join(text.split()).casefold()
            if key in seen:
                continue
            seen.add(key)
            messages.append(text)

        result["post_messages"] = messages
        if messages:
            current_answer = str(result.get("answer") or "").strip()
            parts = [current_answer] if current_answer else []
            parts.extend(messages)
            result["answer"] = "\n\n".join(parts)

        return messages

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
            model_answer = str(result.get("_model_answer") or "").strip()
            pre_rule_count = int(result.get("_pre_rule_batch_count") or 0)

            if (
                reformulate
                and allow_model_reformulation
                and pre_rule_count <= 1
                and model_answer
            ):
                answer = model_answer
            else:
                answer = canonical

            self._append_rule_output(
                result,
                content=answer,
                rule_id=rule.id,
                origin_rule_id=origin_rule_id,
                source=source,
            )
            result["status"] = "answered"
            if not result.get("matched_rule"):
                result["matched_rule"] = origin_rule_id

            await emit_flow(
                emit,
                "rule.action.completed",
                rule_id=rule.id,
                origin_rule_id=origin_rule_id,
                action_type="respond",
                action_index=action_index,
                action_count=action_count,
                status="buffered",
                reformulate=reformulate,
                aggregated_response=True,
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
            post_message = str(
                action.get("post-message", action.get("post_message", "")) or ""
            ).strip()
            if post_message:
                post_messages = result.setdefault("_post_messages", [])
                if not isinstance(post_messages, list):
                    post_messages = []
                    result["_post_messages"] = post_messages
                post_messages.append(post_message)

            ui_action = {
                "type": "suggest_agent",
                "agent": agent_name,
                "label": action.get("label", agent_name.replace("_", " ").title()),
                "arguments": arguments,
                "requires_confirmation": requires_confirmation,
                "rule_id": rule.id,
                "origin_rule_id": origin_rule_id,
            }
            if post_message:
                ui_action["post-message"] = post_message
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
                post_message=post_message or None,
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
                post_message=post_message or None,
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

    async def _handle_grounded_message(
        self,
        user_id: str,
        chat_id: str | None,
        text: str,
        emit: FlowEmitter | None,
        *,
        module: str | None,
        continuation: bool = False,
    ) -> dict[str, Any]:
        """Handle the strict SKB path without exposing general model knowledge.

        Functional rules and arbitrary agents deliberately do not participate in this
        branch: their canonical outputs are not necessarily backed by an SKB page.
        """
        assert self.grounded_answer is not None

        await emit_flow(
            emit,
            "flow.started",
            user_id=user_id,
            requested_chat_id=chat_id,
            mode="skb_grounded",
            module=module,
        )
        if continuation:
            if not chat_id:
                raise ValueError("chat.continue requires an existing chat_id")
            chat = await self.store.get_chat(chat_id, user_id=user_id)
            if chat is None:
                raise ValueError("chat.continue references an unknown chat")
        else:
            chat = await self.store.get_or_create_chat(user_id=user_id, chat_id=chat_id)
        await emit_flow(
            emit,
            "session.ready",
            user_id=user_id,
            chat_id=chat.id,
            existing=bool(chat_id),
            has_codex_thread=False,
        )

        prior_messages = await self.store.list_messages(chat.id)

        if continuation:
            await emit_flow(
                emit,
                "message.user.reused",
                chat_id=chat.id,
                chars=len(text),
                module=module,
            )
        else:
            await self.store.append_message(
                ChatMessage(
                    chat_id=chat.id,
                    role="user",
                    content=text,
                    metadata={"module": module, "grounded": True},
                )
            )
            await emit_flow(
                emit,
                "message.user.persisted",
                chat_id=chat.id,
                chars=len(text),
                module=module,
            )

        try:
            retrieval_question = await self.grounded_answer.rewrite_question(
                text,
                prior_messages,
                module=module,
            )
            await emit_flow(
                emit,
                "rag.retrieval.started",
                module=module,
                source_host=self.grounded_answer.allowed_host,
                reformulated=retrieval_question != " ".join(text.split()).strip(),
            )
            chunks = await self.grounded_answer.retrieve(
                retrieval_question,
                module=module,
            )
            await emit_flow(
                emit,
                "rag.retrieval.completed",
                module=module,
                retrieved_count=len(chunks),
            )

            if chunks:
                await emit_flow(
                    emit,
                    "rag.generation.started",
                    retrieved_count=len(chunks),
                    source_only=True,
                )
            result = await self.grounded_answer.answer_from_chunks(
                retrieval_question,
                chunks,
                module=module,
            )
            result["retrieval_query"] = retrieval_question
            if chunks:
                await emit_flow(
                    emit,
                    "rag.generation.completed",
                    status=result.get("status"),
                    citation_count=len(result.get("sources") or []),
                )
        except Exception as exc:
            await emit_flow(
                emit,
                "rag.failed",
                error=type(exc).__name__,
                module=module,
            )
            result = {
                "status": "source_unavailable",
                "answer": self.grounded_answer.UNAVAILABLE_ANSWER,
                "sources": [],
                "citations": [],
                "module": module,
                "retrieved_count": 0,
                "grounded": True,
                "actions": [],
                "matched_rules": [],
                "matched_rule": None,
            }

        answer = str(result.get("answer") or "")
        sources = [
            {
                "id": source.get("id"),
                "page_id": source.get("page_id"),
                "url": source.get("url"),
                "module": source.get("module"),
                "section": source.get("section"),
                "source_kind": source.get("source_kind"),
                "document_id": source.get("document_id"),
            }
            for source in result.get("sources", [])
            if isinstance(source, dict)
        ]
        await self.store.append_message(
            ChatMessage(
                chat_id=chat.id,
                role="assistant",
                content=answer,
                metadata={
                    "status": result.get("status"),
                    "module": module,
                    "grounded": True,
                    "sources": sources,
                },
            )
        )
        await emit_flow(
            emit,
            "message.assistant.persisted",
            chat_id=chat.id,
            status=result.get("status"),
            chars=len(answer),
            source_count=len(sources),
        )

        result["chat_id"] = chat.id
        await emit_flow(
            emit,
            "flow.completed",
            chat_id=chat.id,
            status=result.get("status"),
            module=module,
            grounded=True,
            source_count=len(sources),
            action_count=len(result.get("actions") or []),
            matched_rule=None,
            matched_rules=[],
            matched_rule_count=0,
            rule_output_count=0,
            post_message_count=0,
        )
        return result

    async def handle_message(
        self,
        user_id: str,
        chat_id: str | None,
        text: str,
        emit: FlowEmitter | None = None,
        *,
        module: str | None = None,
        continuation: bool = False,
    ) -> dict[str, Any]:
        if self.grounded_answer is not None:
            return await self._handle_grounded_message(
                user_id,
                chat_id,
                text,
                emit,
                module=module,
                continuation=continuation,
            )

        if continuation:
            raise ValueError("chat.continue is only available for grounded chat")

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
            multiple_matches_allowed=True,
            timing_scope="integrated_assistant_decision",
        )

        refreshed_chat = await self.store.get_chat(chat.id, user_id)
        await emit_flow(
            emit,
            "reasoning.started",
            available_agent_count=len(self.agents.specs()),
            semantic_rule_count=len(semantic_rules),
            integrated_rule_matching=True,
            multiple_rule_matches=True,
        )

        result = await self.codex.answer_with_rules(
            user_message=text,
            rendered_context=rendered_context,
            rules=[rule.model_dump() for rule in semantic_rules],
            agents=self.agents.specs(),
            thread_id=refreshed_chat.codex_thread_id if refreshed_chat else None,
            emit=emit,
        )
        result["_model_answer"] = result.get("answer")
        result["_rule_outputs"] = []

        rule_decision_elapsed_ms = round((perf_counter() - rule_decision_started_at) * 1000, 1)
        proposed_matches = result.get("matched_rules")
        if not isinstance(proposed_matches, list):
            proposed_matches = []
        proposed_rule_ids = [
            str(item.get("rule_id") or "")
            for item in proposed_matches
            if isinstance(item, dict) and item.get("rule_id")
        ]

        pre_rules = await self.rules.resolve_pre_decisions(
            result,
            emit=emit,
            decision_elapsed_ms=rule_decision_elapsed_ms,
        )

        if proposed_rule_ids and not pre_rules:
            result.update(
                {
                    "matched_rules": [],
                    "matched_rule": None,
                    "rule_confidence": None,
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
                proposed_rule_ids=proposed_rule_ids,
                reason="all_proposed_rules_rejected",
            )

        if result.get("thread_id") and result.get("thread_id") != chat.codex_thread_id:
            await self.store.set_codex_thread(chat.id, result["thread_id"])
            await emit_flow(
                emit,
                "session.codex_thread.updated",
                chat_id=chat.id,
                thread_id=result["thread_id"],
            )

        if pre_rules:
            confidence_by_id = {
                str(item.get("rule_id") or ""): item.get("confidence")
                for item in proposed_matches
                if isinstance(item, dict)
            }
            accepted_matches = [
                {"rule_id": rule.id, "confidence": confidence_by_id.get(rule.id)}
                for rule in pre_rules
            ]

            result["actions"] = []
            result["suggested_agent"] = None
            result["suggested_agent_args"] = {}
            result["matched_rules"] = accepted_matches
            result["matched_rule"] = pre_rules[0].id
            result["rule_confidence"] = accepted_matches[0]["confidence"]
            result["_pre_rule_batch_count"] = len(pre_rules)

            for rule_index, pre_rule in enumerate(pre_rules):
                await emit_flow(
                    emit,
                    "rules.pre.execution.started",
                    rule_id=pre_rule.id,
                    rule_index=rule_index,
                    rule_count=len(pre_rules),
                    priority=pre_rule.priority,
                )
                success = await self._apply_rule_actions(
                    pre_rule,
                    result,
                    emit,
                    source="pre_rule",
                    allow_model_reformulation=True,
                    origin_rule_id=pre_rule.id,
                )
                await emit_flow(
                    emit,
                    "rules.pre.execution.completed" if success else "rules.pre.execution.failed",
                    rule_id=pre_rule.id,
                    rule_index=rule_index,
                    rule_count=len(pre_rules),
                    priority=pre_rule.priority,
                )
        else:
            result.setdefault("matched_rules", [])
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
            matched_rules=[
                item.get("rule_id")
                for item in result.get("matched_rules", [])
                if isinstance(item, dict)
            ],
            matched_rule_count=len(result.get("matched_rules") or []),
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

        rule_outputs = self._compose_rule_outputs(result)
        if rule_outputs:
            await emit_flow(
                emit,
                "response.rules.composed",
                output_count=len(rule_outputs),
                rule_ids=[item["origin_rule_id"] for item in rule_outputs],
                chars=len(str(result.get("answer") or "")),
                separator="blank_line",
                deduplicated=True,
            )

        post_messages = self._compose_post_messages(result)
        if post_messages:
            await emit_flow(
                emit,
                "response.post_messages.composed",
                message_count=len(post_messages),
                chars=sum(len(message) for message in post_messages),
                separator="blank_line",
                status_preserved=True,
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

        result.pop("_model_answer", None)
        result.pop("_pre_rule_batch_count", None)

        matched_rule_ids = [
            item.get("rule_id")
            for item in result.get("matched_rules", [])
            if isinstance(item, dict)
        ]
        await self.store.append_message(
            ChatMessage(
                chat_id=chat.id,
                role="assistant",
                content=str(answer or ""),
                metadata={
                    "status": result.get("status"),
                    "matched_rule": result.get("matched_rule"),
                    "matched_rules": matched_rule_ids,
                    "rule_outputs": [
                        {
                            "rule_id": item.get("origin_rule_id"),
                            "source": item.get("source"),
                        }
                        for item in result.get("rule_outputs", [])
                        if isinstance(item, dict)
                    ],
                    "post_messages": result.get("post_messages", []),
                },
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
            matched_rules=matched_rule_ids,
            matched_rule_count=len(matched_rule_ids),
            rule_output_count=len(result.get("rule_outputs") or []),
            post_message_count=len(result.get("post_messages") or []),
        )
        return result
