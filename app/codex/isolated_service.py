from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from app.flow.events import FlowEmitter, emit_flow

from .client import CodexResult
from .service import CodexService


class IsolatedDecisionService(CodexService):
    """Two-stage native Ollama decision pipeline.

    Stage 1 classifies semantic rules from the latest user message ONLY. Conversation
    history is deliberately absent, so a previous rule intent cannot leak into a follow-up.

    Stage 2 performs answer reasoning with conversation context. Rule matches are already
    frozen and the answer schema contains no rule-classification fields, so this stage cannot
    add, remove, or change matched rules.
    """

    @staticmethod
    def _classifier_schema(valid_rule_ids: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "matched_rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rule_id": {"type": "string", "enum": valid_rule_ids},
                            "confidence": {
                                "anyOf": [
                                    {"type": "number", "minimum": 0, "maximum": 1},
                                    {"type": "null"},
                                ]
                            },
                        },
                        "required": ["rule_id", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "has_uncovered_request": {"type": "boolean"},
            },
            "required": ["matched_rules", "has_uncovered_request"],
            "additionalProperties": False,
        }

    @classmethod
    def _answer_schema(cls, valid_agent_names: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["answered", "not_found", "insufficient_information"],
                },
                "answer": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
                "suggested_agent": cls._nullable_enum(valid_agent_names),
                "suggested_agent_args": {"type": "object"},
            },
            "required": ["status", "answer", "suggested_agent", "suggested_agent_args"],
            "additionalProperties": False,
        }

    @staticmethod
    def _contains_reformulatable_respond(node: Any) -> bool:
        if isinstance(node, list):
            return any(IsolatedDecisionService._contains_reformulatable_respond(item) for item in node)
        if not isinstance(node, dict):
            return False
        if str(node.get("type") or "").strip() == "respond" and bool(node.get("reformulate", False)):
            return True
        return (
            IsolatedDecisionService._contains_reformulatable_respond(node.get("then"))
            or IsolatedDecisionService._contains_reformulatable_respond(node.get("catch"))
        )

    async def _native_json_operation(
        self,
        *,
        operation: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        thread_id: str | None,
        emit: FlowEmitter | None,
    ) -> CodexResult:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        prompt_tokens_estimated = self._estimate_tokens(combined_prompt)

        if self.decision_client is None:
            return await self.complete(
                combined_prompt,
                thread_id=thread_id,
                operation=operation,
                thinking_mode="uncontrolled",
                emit=emit,
            )

        model = self.decision_client.model or self.client.model or "qwen3:8b"
        await emit_flow(
            emit,
            "model.request.started",
            provider="ollama_native",
            operation=operation,
            model=model,
            thinking_mode="disabled",
            thinking_control="native_think_false",
            prompt_layout="system_user",
            structured_output="json_schema",
            temperature=0,
            thread_reused=False,
            prompt_chars=len(combined_prompt),
            prompt_tokens_estimated=prompt_tokens_estimated,
            estimated_tokens=prompt_tokens_estimated,
            timing_scope="ollama_native_api",
        )

        started_at = perf_counter()
        try:
            native = await self.decision_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                format_schema=schema,
                model=model,
                think=False,
                temperature=0.0,
            )
        except Exception as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 1)
            await emit_flow(
                emit,
                "model.request.failed",
                provider="ollama_native",
                operation=operation,
                model=model,
                thinking_mode="disabled",
                thinking_control="native_think_false",
                elapsed_ms=elapsed_ms,
                elapsed_seconds=round(elapsed_ms / 1000, 3),
                timing_scope="ollama_native_api",
                error=type(exc).__name__,
            )
            raise

        wall_elapsed_ms = round((perf_counter() - started_at) * 1000, 1)
        total_s = self._seconds(native.total_duration_ns)
        prompt_eval_s = self._seconds(native.prompt_eval_duration_ns)
        eval_s = self._seconds(native.eval_duration_ns)
        prompt_tok_s = self._tokens_per_second(native.prompt_eval_count, native.prompt_eval_duration_ns)
        output_tok_s = self._tokens_per_second(native.eval_count, native.eval_duration_ns)
        output_tokens_estimated = self._estimate_tokens(native.text)
        displayed_total = total_s if total_s is not None else wall_elapsed_ms / 1000

        await emit_flow(
            emit,
            "model.request.completed",
            provider="ollama_native",
            operation=operation,
            model=model,
            thinking_mode="disabled",
            thinking_control="native_think_false",
            prompt_layout="system_user",
            structured_output="json_schema",
            temperature=0,
            prompt_chars=len(combined_prompt),
            output_chars=len(native.text),
            prompt_tokens_estimated=prompt_tokens_estimated,
            output_tokens_estimated=output_tokens_estimated,
            elapsed_ms=wall_elapsed_ms,
            elapsed_seconds=round(wall_elapsed_ms / 1000, 3),
            ollama_total_seconds=total_s,
            ollama_prompt_eval_count=native.prompt_eval_count,
            ollama_prompt_eval_seconds=prompt_eval_s,
            ollama_prompt_tokens_per_second=prompt_tok_s,
            ollama_eval_count=native.eval_count,
            ollama_eval_seconds=eval_s,
            ollama_output_tokens_per_second=output_tok_s,
            timing_scope="ollama_native_api",
            metrics_source="ollama_native_response",
            native_ollama_eval_metrics_available=True,
            status=(
                f"{model} · {displayed_total:.1f} s · "
                f"prompt={native.prompt_eval_count if native.prompt_eval_count is not None else f'≈{prompt_tokens_estimated}'} tok"
                + (f" @ {prompt_tok_s:.1f} tok/s" if prompt_tok_s is not None else "")
                + f" · output={native.eval_count if native.eval_count is not None else f'≈{output_tokens_estimated}'} tok"
                + (f" @ {output_tok_s:.1f} tok/s" if output_tok_s is not None else "")
                + " · think:false · schema"
            ),
            estimated_tokens=(native.prompt_eval_count or prompt_tokens_estimated),
        )
        return CodexResult(text=native.text, thread_id=thread_id, raw=native.raw)

    async def answer_with_rules(
        self,
        user_message: str,
        rendered_context: str,
        rules: list[dict],
        agents: list[dict],
        thread_id: str | None,
        emit: FlowEmitter | None = None,
    ) -> dict:
        compact_rules = [
            {
                "id": str(rule["id"]),
                "priority": int(rule.get("priority", 0)),
                "description": str(rule.get("description", "")),
                "when": rule.get("when", {}),
            }
            for rule in rules
        ]
        valid_rule_ids = [rule["id"] for rule in compact_rules]
        rules_by_id = {str(rule["id"]): rule for rule in rules}
        valid_agent_names = [
            str(agent.get("name") or agent.get("id") or "").strip()
            for agent in agents
            if str(agent.get("name") or agent.get("id") or "").strip()
        ]

        classifier_system = f"""
You are a semantic functional-rule classifier.

ABSOLUTE ISOLATION RULES:
- Classify ONLY the latest user message supplied below.
- You have no conversation history and must not assume any previous user intent.
- Compare every clause/request in the latest message independently with every rule.
- Return every applicable rule, ordered by priority descending.
- `has_uncovered_request` is true when any request/question in the latest message is not covered
  by the matched rules. If no rule matches, it is true for any non-empty user request.
- Similar vocabulary or an entity associated with a rule is NOT enough: the current message must
  actually request the operation described by that rule.

SEMANTIC RULES:
{json.dumps(compact_rules, ensure_ascii=False)}
""".strip()
        classifier_user = f"LATEST USER MESSAGE ONLY:\n{user_message}".strip()

        classifier_result = await self._native_json_operation(
            operation="rule_classifier",
            system_prompt=classifier_system,
            user_prompt=classifier_user,
            schema=self._classifier_schema(valid_rule_ids),
            thread_id=thread_id,
            emit=emit,
        )
        classifier_payload = self._json_object(classifier_result.text, {})
        matches, parse_mode = self._normalize_matched_rules(classifier_payload, valid_rule_ids)
        has_uncovered_request = bool(classifier_payload.get("has_uncovered_request", not matches))

        await emit_flow(
            emit,
            "rules.pre.decision_parsed",
            rule_id=matches[0]["rule_id"] if matches else None,
            rule_ids=[match["rule_id"] for match in matches],
            match_count=len(matches),
            confidence=matches[0]["confidence"] if matches else None,
            parse_mode=parse_mode,
            classifier_context_isolated=True,
            classifier_context_tokens=0,
            has_uncovered_request=has_uncovered_request,
            thinking_mode="disabled",
            thinking_control="native_think_false" if self.decision_client else "uncontrolled",
            structured_output="json_schema" if self.decision_client else "prompt_only",
        )

        matched_rule_defs = [
            rules_by_id[match["rule_id"]]
            for match in matches
            if match["rule_id"] in rules_by_id
        ]
        needs_reformulation = any(
            self._contains_reformulatable_respond(rule.get("then"))
            for rule in matched_rule_defs
        )
        needs_answer_reasoning = not matches or has_uncovered_request or needs_reformulation

        answer_payload: dict[str, Any] = {
            "status": "answered" if matches else "insufficient_information",
            "answer": None,
            "suggested_agent": None,
            "suggested_agent_args": {},
        }

        if needs_answer_reasoning:
            selected_rules = [
                {
                    "id": str(rule["id"]),
                    "description": str(rule.get("description", "")),
                    "then": rule.get("then", []),
                }
                for rule in matched_rule_defs
            ]
            answer_system = f"""
You are the answer-reasoning stage of an assistant.
Rule classification has ALREADY finished in a separate context-isolated stage.
You MUST NOT classify rules, add rules, remove rules, or change the matched rule list.

FROZEN MATCHED RULES:
{json.dumps(selected_rules, ensure_ascii=False)}

ANSWER REQUIREMENTS:
- Answer the entire latest user message naturally.
- Use conversation context for factual/coreference continuity only.
- If a frozen rule requires a response outcome, preserve that required outcome.
- Also answer every request not covered by the frozen rules.
- Do not invent unavailable facts. Use not_found/insufficient_information when appropriate.
- You may suggest at most one available read-only/code agent when useful, but never claim it ran.

AVAILABLE CODE AGENTS:
{json.dumps(agents, ensure_ascii=False)}
""".strip()
            answer_user = f"""
LATEST USER MESSAGE:
{user_message}

CONVERSATION CONTEXT:
{rendered_context or '(none)'}
""".strip()
            answer_result = await self._native_json_operation(
                operation="answer_reasoning",
                system_prompt=answer_system,
                user_prompt=answer_user,
                schema=self._answer_schema(valid_agent_names),
                thread_id=thread_id,
                emit=emit,
            )
            parsed_answer = self._json_object(answer_result.text, {})
            if parsed_answer:
                status = parsed_answer.get("status")
                if status not in {"answered", "not_found", "insufficient_information"}:
                    status = "answered" if parsed_answer.get("answer") else "insufficient_information"
                answer = parsed_answer.get("answer")
                if answer is not None and not isinstance(answer, str):
                    answer = str(answer)
                suggested_agent = parsed_answer.get("suggested_agent")
                if suggested_agent not in valid_agent_names:
                    suggested_agent = None
                suggested_args = parsed_answer.get("suggested_agent_args")
                if not isinstance(suggested_args, dict):
                    suggested_args = {}
                answer_payload = {
                    "status": status,
                    "answer": answer,
                    "suggested_agent": suggested_agent,
                    "suggested_agent_args": suggested_args,
                }

        primary = matches[0] if matches else None
        return {
            "matched_rules": matches,
            "matched_rule": primary["rule_id"] if primary else None,
            "rule_confidence": primary["confidence"] if primary else None,
            "status": answer_payload["status"],
            "answer": answer_payload["answer"],
            "suggested_agent": answer_payload["suggested_agent"],
            "suggested_agent_args": answer_payload["suggested_agent_args"],
            "parse_mode": parse_mode,
            "has_uncovered_request": has_uncovered_request,
            "decision_pipeline": "isolated_classifier_then_answer",
            "thread_id": thread_id,
        }
