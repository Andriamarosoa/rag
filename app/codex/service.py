from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from app.context.manager import Summarizer
from app.flow.events import FlowEmitter, emit_flow
from app.sessions.models import ChatMessage

from .client import CodexMcpClient, CodexResult


class CodexService(Summarizer):
    def __init__(self, client: CodexMcpClient):
        self.client = client

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Cheap model-agnostic estimate used only for UI/performance diagnostics."""
        return max(1, len(text) // 4) if text else 0

    async def complete(
        self,
        prompt: str,
        thread_id: str | None = None,
        *,
        operation: str = "complete",
        thinking_mode: str | None = None,
        emit: FlowEmitter | None = None,
    ) -> CodexResult:
        prompt_tokens_estimated = self._estimate_tokens(prompt)
        await emit_flow(
            emit,
            "model.request.started",
            provider="codex_ollama",
            operation=operation,
            model=self.client.model or None,
            thinking_mode=thinking_mode,
            thread_reused=bool(thread_id),
            prompt_chars=len(prompt),
            prompt_tokens_estimated=prompt_tokens_estimated,
            estimated_tokens=prompt_tokens_estimated,
            timing_scope="codex_to_ollama_round_trip",
        )

        started_at = perf_counter()
        try:
            result = await self.client.ask(prompt, thread_id=thread_id)
        except Exception as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 1)
            await emit_flow(
                emit,
                "model.request.failed",
                provider="codex_ollama",
                operation=operation,
                model=self.client.model or None,
                thinking_mode=thinking_mode,
                elapsed_ms=elapsed_ms,
                elapsed_seconds=round(elapsed_ms / 1000, 3),
                timing_scope="codex_to_ollama_round_trip",
                error=type(exc).__name__,
            )
            raise

        elapsed_ms = round((perf_counter() - started_at) * 1000, 1)
        output_tokens_estimated = self._estimate_tokens(result.text)
        await emit_flow(
            emit,
            "model.request.completed",
            provider="codex_ollama",
            operation=operation,
            model=self.client.model or None,
            thinking_mode=thinking_mode,
            thread_id=result.thread_id,
            prompt_chars=len(prompt),
            output_chars=len(result.text),
            prompt_tokens_estimated=prompt_tokens_estimated,
            output_tokens_estimated=output_tokens_estimated,
            elapsed_ms=elapsed_ms,
            elapsed_seconds=round(elapsed_ms / 1000, 3),
            timing_scope="codex_to_ollama_round_trip",
            metrics_source="application_wall_clock",
            native_ollama_eval_metrics_available=False,
            status=(
                f"{self.client.model or 'model'} · {elapsed_ms / 1000:.1f} s · "
                f"prompt≈{prompt_tokens_estimated} tok · output≈{output_tokens_estimated} tok"
                + (f" · {thinking_mode}" if thinking_mode else "")
            ),
            estimated_tokens=prompt_tokens_estimated,
        )
        return result

    async def summarize_context(
        self,
        previous_summary: str,
        messages: list[ChatMessage],
        emit: FlowEmitter | None = None,
    ) -> str:
        transcript = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
        prompt = f"""
You are compacting a long-running application chat context.
Preserve facts, user preferences stated in the chat, decisions, unresolved questions,
important identifiers, tool outcomes, and constraints. Remove repetition and chit-chat.
Do not invent facts. Return only the new rolling summary in plain text.

PREVIOUS SUMMARY:
{previous_summary or '(none)'}

MESSAGES TO COMPACT:
{transcript}
""".strip()
        result = await self.complete(prompt, operation="context_summarization", emit=emit)
        return result.text.strip()

    async def answer_with_rules(
        self,
        user_message: str,
        rendered_context: str,
        rules: list[dict],
        agents: list[dict],
        thread_id: str | None,
        emit: FlowEmitter | None = None,
    ) -> dict:
        """Classify semantic rules and reason about the answer in one model request."""
        compact_rules = [
            {
                "id": rule["id"],
                "priority": rule.get("priority", 0),
                "description": rule.get("description", ""),
                "when": rule.get("when", {}),
                "then": rule.get("then", {}),
            }
            for rule in rules
        ]
        valid_rule_ids = [str(rule["id"]) for rule in compact_rules]

        prompt = f"""
You are the local reasoning engine behind a WebSocket assistant.
Perform rule classification AND answer reasoning in this single pass.

TASK 1 — FUNCTIONAL RULES
Classify only the LATEST USER MESSAGE against the semantic functional rules below.
Rules describe meanings, not literal phrases. Select the single highest-priority rule whose
meaning clearly applies. Otherwise matched_rule must be null.

When a selected rule has then.type = "respond":
- status MUST be "answered".
- If reformulate=false, answer MUST equal canonical_answer exactly.
- If reformulate=true, naturally rephrase canonical_answer for the user, but add no new facts.
- Do not suggest an agent unless the selected rule explicitly requires one.

TASK 2 — NORMAL ANSWER
If no pre-rule applies, answer using the available context. If the context is insufficient,
do not invent an answer. You may suggest one available code-defined agent, but DO NOT claim
it has executed.

Return exactly ONE JSON object and nothing else with this shape:
{{
  "matched_rule": string | null,
  "rule_confidence": number | null,
  "status": "answered" | "not_found" | "insufficient_information",
  "answer": string | null,
  "suggested_agent": string | null,
  "suggested_agent_args": object
}}

VALID RULE IDS:
{json.dumps(valid_rule_ids, ensure_ascii=False)}

SEMANTIC PRE-RULES:
{json.dumps(compact_rules, ensure_ascii=False)}

AVAILABLE CODE AGENTS:
{json.dumps(agents, ensure_ascii=False)}

CONTEXT:
{rendered_context}

LATEST USER MESSAGE:
{user_message}

/no_think
""".strip()

        result = await self.complete(
            prompt,
            thread_id=thread_id,
            operation="assistant_decision",
            thinking_mode="no_think",
            emit=emit,
        )
        payload = self._parse_assistant_decision(result.text, valid_rule_ids)
        payload["thread_id"] = result.thread_id
        await emit_flow(
            emit,
            "rules.pre.decision_parsed",
            rule_id=payload.get("matched_rule"),
            confidence=payload.get("rule_confidence"),
            parse_mode=payload.get("parse_mode"),
            thinking_mode="no_think",
        )
        return payload

    @staticmethod
    def _normalize_confidence(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        if 1 < confidence <= 100:
            confidence /= 100
        if not 0 <= confidence <= 1:
            return None
        return confidence

    @classmethod
    def _parse_assistant_decision(cls, text: str, valid_rule_ids: list[str]) -> dict:
        raw = (text or "").strip()
        valid = set(valid_rule_ids)
        parsed = cls._json_object(raw, {})

        if parsed:
            rule_id = parsed.get("matched_rule", parsed.get("rule_id"))
            if isinstance(rule_id, str):
                rule_id = rule_id.strip()
                if rule_id.lower() in {"null", "none", "no_match", "nomatch"}:
                    rule_id = None
            parse_mode = "json"
            if rule_id is not None and rule_id not in valid:
                rule_id = None
                parse_mode = "json_unknown_rule"

            status = parsed.get("status")
            if status not in {"answered", "not_found", "insufficient_information"}:
                status = "answered" if parsed.get("answer") else "insufficient_information"

            answer = parsed.get("answer")
            if answer is not None and not isinstance(answer, str):
                answer = str(answer)

            suggested_agent = parsed.get("suggested_agent")
            if suggested_agent is not None and not isinstance(suggested_agent, str):
                suggested_agent = str(suggested_agent)

            suggested_args = parsed.get("suggested_agent_args")
            if not isinstance(suggested_args, dict):
                suggested_args = {}

            return {
                "matched_rule": rule_id,
                "rule_confidence": cls._normalize_confidence(
                    parsed.get("rule_confidence", parsed.get("confidence"))
                ),
                "status": status,
                "answer": answer,
                "suggested_agent": suggested_agent,
                "suggested_agent_args": suggested_args,
                "parse_mode": parse_mode,
            }

        rule_decision = cls._parse_rule_decision(raw, valid_rule_ids)
        matched_rule = rule_decision.get("rule_id")
        return {
            "matched_rule": matched_rule,
            "rule_confidence": rule_decision.get("confidence"),
            "status": "answered" if matched_rule else "insufficient_information",
            "answer": None,
            "suggested_agent": None,
            "suggested_agent_args": {},
            "parse_mode": rule_decision.get("parse_mode"),
        }

    @classmethod
    def _parse_rule_decision(cls, text: str, valid_rule_ids: list[str]) -> dict:
        """Parse an explicit rule-only decision used as a tolerant fallback."""
        raw = (text or "").strip()
        valid = set(valid_rule_ids)

        parsed = cls._json_object(raw, {})
        if parsed:
            rule_id = parsed.get("rule_id", parsed.get("matched_rule"))
            if isinstance(rule_id, str):
                rule_id = rule_id.strip()
                if rule_id.lower() in {"null", "none", "no_match", "nomatch"}:
                    rule_id = None
            if rule_id is None or rule_id in valid:
                return {
                    "rule_id": rule_id,
                    "confidence": cls._normalize_confidence(
                        parsed.get("confidence", parsed.get("rule_confidence"))
                    ),
                    "parse_mode": "json",
                }

        rule_match = re.search(
            r"\brule[_\s-]*id\s*[:=]\s*[\"']?([A-Za-z0-9_.-]+|null|none)[\"']?",
            raw,
            re.IGNORECASE,
        )
        if rule_match:
            candidate = rule_match.group(1).strip()
            rule_id = None if candidate.lower() in {"null", "none"} else candidate
            if rule_id is None or rule_id in valid:
                confidence_match = re.search(
                    r"\bconfidence\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
                    raw,
                    re.IGNORECASE,
                )
                confidence = cls._normalize_confidence(
                    confidence_match.group(1) if confidence_match else None
                )
                return {
                    "rule_id": rule_id,
                    "confidence": confidence,
                    "parse_mode": "key_value",
                }

        compact = raw.strip().strip("`\"'").strip()
        if compact in valid:
            return {
                "rule_id": compact,
                "confidence": None,
                "parse_mode": "exact_rule_id",
            }
        if compact.lower() in {"null", "none", "no_match", "nomatch"}:
            return {
                "rule_id": None,
                "confidence": None,
                "parse_mode": "exact_none",
            }

        return {
            "rule_id": None,
            "confidence": None,
            "parse_mode": "unparseable",
        }

    @staticmethod
    def _json_object(text: str, fallback: dict) -> dict:
        text = text.strip()
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else fallback
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else fallback
            except Exception:
                pass
        return fallback