from __future__ import annotations

import json
import re
from typing import Any

from app.context.manager import Summarizer
from app.flow.events import FlowEmitter, emit_flow
from app.sessions.models import ChatMessage

from .client import CodexMcpClient, CodexResult


class CodexService(Summarizer):
    def __init__(self, client: CodexMcpClient):
        self.client = client

    async def complete(
        self,
        prompt: str,
        thread_id: str | None = None,
        *,
        operation: str = "complete",
        emit: FlowEmitter | None = None,
    ) -> CodexResult:
        await emit_flow(
            emit,
            "model.request.started",
            provider="codex",
            operation=operation,
            thread_reused=bool(thread_id),
            prompt_chars=len(prompt),
        )
        try:
            result = await self.client.ask(prompt, thread_id=thread_id)
        except Exception as exc:
            await emit_flow(
                emit,
                "model.request.failed",
                provider="codex",
                operation=operation,
                error=type(exc).__name__,
            )
            raise
        await emit_flow(
            emit,
            "model.request.completed",
            provider="codex",
            operation=operation,
            thread_id=result.thread_id,
            output_chars=len(result.text),
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

    async def choose_pre_rule(
        self,
        user_message: str,
        rendered_context: str,
        rules: list[dict],
        emit: FlowEmitter | None = None,
    ) -> dict:
        compact_rules = [
            {
                "id": r["id"],
                "description": r.get("description", ""),
                "when": r.get("when", {}),
                "then": r.get("then", {}),
            }
            for r in rules
        ]
        valid_rule_ids = [str(r["id"]) for r in compact_rules]
        prompt = f"""
Classify the user's LATEST MESSAGE against the functional rules below.
Rules describe semantic meanings, not literal phrases. Select a rule only when its meaning
clearly applies. Otherwise select null.

Return exactly ONE JSON object and nothing else:
{{"rule_id":"<one valid id or null>","confidence":<number from 0 to 1>}}
Do not use markdown. Do not explain the decision. Do not repeat the rules.

VALID RULE IDS:
{json.dumps(valid_rule_ids, ensure_ascii=False)}

RULES:
{json.dumps(compact_rules, ensure_ascii=False)}

CONTEXT (background only; classify the latest message):
{rendered_context}

LATEST USER MESSAGE:
{user_message}
""".strip()
        result = await self.complete(prompt, operation="pre_rule_matching", emit=emit)
        decision = self._parse_rule_decision(result.text, valid_rule_ids)
        await emit_flow(
            emit,
            "rules.pre.decision_parsed",
            rule_id=decision.get("rule_id"),
            confidence=decision.get("confidence"),
            parse_mode=decision.get("parse_mode"),
        )
        return decision

    async def reformulate(
        self,
        canonical_answer: str,
        user_message: str,
        rendered_context: str,
        emit: FlowEmitter | None = None,
    ) -> str:
        prompt = f"""
Rephrase the canonical answer naturally for the current user message.
Do not add any fact not contained in the canonical answer.
Return only the final answer.

CANONICAL ANSWER:
{canonical_answer}

USER MESSAGE:
{user_message}

CONTEXT:
{rendered_context}
""".strip()
        return (
            await self.complete(prompt, operation="rule_answer_reformulation", emit=emit)
        ).text.strip()

    async def answer(
        self,
        user_message: str,
        rendered_context: str,
        agents: list[dict],
        thread_id: str | None,
        emit: FlowEmitter | None = None,
    ) -> dict:
        prompt = f"""
You are the local reasoning engine behind a WebSocket assistant.
Answer using the available context. If the context is insufficient, do not invent an answer.
You may suggest one of the code-defined agents, but DO NOT claim it has executed.

Return STRICT JSON only with this shape:
{{
  "status": "answered" | "not_found" | "insufficient_information",
  "answer": string | null,
  "suggested_agent": string | null,
  "suggested_agent_args": object
}}

AVAILABLE CODE AGENTS:
{json.dumps(agents, ensure_ascii=False)}

CONTEXT:
{rendered_context}

LATEST USER MESSAGE:
{user_message}
""".strip()
        result = await self.complete(
            prompt,
            thread_id=thread_id,
            operation="assistant_reasoning",
            emit=emit,
        )
        payload = self._json_object(
            result.text,
            {
                "status": "insufficient_information",
                "answer": None,
                "suggested_agent": None,
                "suggested_agent_args": {},
            },
        )
        payload["thread_id"] = result.thread_id
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
    def _parse_rule_decision(cls, text: str, valid_rule_ids: list[str]) -> dict:
        """Parse a small classifier response without silently turning formatting noise into no-match.

        Models used behind Codex/Ollama do not all obey JSON-only output equally well. We still
        require the rule id to be an exact configured id; tolerant parsing only accepts explicit
        decision-shaped output, never an arbitrary occurrence of a rule id in prose.
        """
        raw = (text or "").strip()
        valid = set(valid_rule_ids)

        parsed = cls._json_object(raw, {})
        if parsed:
            rule_id = parsed.get("rule_id")
            if isinstance(rule_id, str):
                rule_id = rule_id.strip()
                if rule_id.lower() in {"null", "none", "no_match", "nomatch"}:
                    rule_id = None
            if rule_id is None or rule_id in valid:
                return {
                    "rule_id": rule_id,
                    "confidence": cls._normalize_confidence(parsed.get("confidence")),
                    "parse_mode": "json",
                }

        # Accept explicit key/value output such as:
        # rule_id=password_reset confidence=0.94
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

        # Some small local models return exactly the selected id despite the requested JSON.
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
