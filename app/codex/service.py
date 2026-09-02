from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from app.context.manager import Summarizer
from app.flow.events import FlowEmitter, emit_flow
from app.ollama.client import OllamaNativeClient
from app.sessions.models import ChatMessage

from .client import CodexMcpClient, CodexResult


class CodexService(Summarizer):
    def __init__(
        self,
        client: CodexMcpClient,
        decision_client: OllamaNativeClient | None = None,
    ):
        self.client = client
        self.decision_client = decision_client

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Cheap model-agnostic estimate used only for UI/performance diagnostics."""
        return max(1, len(text) // 4) if text else 0

    @staticmethod
    def _seconds(ns: int | None) -> float | None:
        return round(ns / 1_000_000_000, 6) if ns is not None else None

    @staticmethod
    def _tokens_per_second(count: int | None, duration_ns: int | None) -> float | None:
        if count is None or duration_ns is None or duration_ns <= 0:
            return None
        return round(count / (duration_ns / 1_000_000_000), 2)

    @staticmethod
    def _nullable_enum(values: list[str]) -> dict[str, Any]:
        if not values:
            return {"type": "null"}
        return {
            "anyOf": [
                {"type": "string", "enum": values},
                {"type": "null"},
            ]
        }

    @classmethod
    def _decision_schema(
        cls,
        valid_rule_ids: list[str],
        valid_agent_names: list[str],
    ) -> dict[str, Any]:
        matched_rule_item: dict[str, Any] = {
            "type": "object",
            "properties": {
                "rule_id": {
                    "type": "string",
                    "enum": valid_rule_ids,
                },
                "confidence": {
                    "anyOf": [
                        {"type": "number", "minimum": 0, "maximum": 1},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["rule_id", "confidence"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "matched_rules": {
                    "type": "array",
                    "items": matched_rule_item,
                },
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
            "required": [
                "matched_rules",
                "status",
                "answer",
                "suggested_agent",
                "suggested_agent_args",
            ],
            "additionalProperties": False,
        }

    async def complete(
        self,
        prompt: str,
        thread_id: str | None = None,
        *,
        operation: str = "complete",
        thinking_mode: str | None = None,
        emit: FlowEmitter | None = None,
    ) -> CodexResult:
        """Codex-backed model call kept for operations that should use the Codex harness."""
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

    async def _complete_assistant_decision(
        self,
        system_prompt: str,
        user_prompt: str,
        valid_rule_ids: list[str],
        valid_agent_names: list[str],
        thread_id: str | None,
        emit: FlowEmitter | None,
    ) -> CodexResult:
        """Fast deterministic path using Ollama native API with real `think=false`."""
        combined_prompt = f"{system_prompt}\n\n{user_prompt}".strip()

        if self.decision_client is None:
            return await self.complete(
                combined_prompt,
                thread_id=thread_id,
                operation="assistant_decision",
                emit=emit,
            )

        model = self.decision_client.model or self.client.model or "qwen3:8b"
        prompt_tokens_estimated = self._estimate_tokens(combined_prompt)
        await emit_flow(
            emit,
            "model.request.started",
            provider="ollama_native",
            operation="assistant_decision",
            model=model,
            thinking_mode="disabled",
            thinking_control="native_think_false",
            prompt_layout="system_user",
            structured_output="json_schema",
            temperature=0,
            thread_reused=bool(thread_id),
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
                format_schema=self._decision_schema(valid_rule_ids, valid_agent_names),
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
                operation="assistant_decision",
                model=model,
                thinking_mode="disabled",
                thinking_control="native_think_false",
                prompt_layout="system_user",
                structured_output="json_schema",
                temperature=0,
                elapsed_ms=elapsed_ms,
                elapsed_seconds=round(elapsed_ms / 1000, 3),
                timing_scope="ollama_native_api",
                error=type(exc).__name__,
            )
            raise

        wall_elapsed_ms = round((perf_counter() - started_at) * 1000, 1)
        total_s = self._seconds(native.total_duration_ns)
        load_s = self._seconds(native.load_duration_ns)
        prompt_eval_s = self._seconds(native.prompt_eval_duration_ns)
        eval_s = self._seconds(native.eval_duration_ns)
        prompt_tok_s = self._tokens_per_second(
            native.prompt_eval_count,
            native.prompt_eval_duration_ns,
        )
        output_tok_s = self._tokens_per_second(native.eval_count, native.eval_duration_ns)
        output_tokens_estimated = self._estimate_tokens(native.text)

        prompt_count_label = (
            native.prompt_eval_count
            if native.prompt_eval_count is not None
            else f"≈{prompt_tokens_estimated}"
        )
        output_count_label = (
            native.eval_count
            if native.eval_count is not None
            else f"≈{output_tokens_estimated}"
        )
        prompt_speed_label = f" @ {prompt_tok_s:.1f} tok/s" if prompt_tok_s is not None else ""
        output_speed_label = f" @ {output_tok_s:.1f} tok/s" if output_tok_s is not None else ""
        displayed_total = total_s if total_s is not None else wall_elapsed_ms / 1000

        await emit_flow(
            emit,
            "model.request.completed",
            provider="ollama_native",
            operation="assistant_decision",
            model=model,
            thinking_mode="disabled",
            thinking_control="native_think_false",
            prompt_layout="system_user",
            structured_output="json_schema",
            temperature=0,
            thread_id=thread_id,
            prompt_chars=len(combined_prompt),
            output_chars=len(native.text),
            prompt_tokens_estimated=prompt_tokens_estimated,
            output_tokens_estimated=output_tokens_estimated,
            elapsed_ms=wall_elapsed_ms,
            elapsed_seconds=round(wall_elapsed_ms / 1000, 3),
            ollama_total_seconds=total_s,
            ollama_load_seconds=load_s,
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
                f"prompt={prompt_count_label} tok{prompt_speed_label} · "
                f"output={output_count_label} tok{output_speed_label} · "
                "think:false · schema"
            ),
            estimated_tokens=(native.prompt_eval_count or prompt_tokens_estimated),
        )
        return CodexResult(text=native.text, thread_id=thread_id, raw=native.raw)

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
        """Classify all applicable semantic rules and reason about the answer in one pass."""
        compact_rules = [
            {
                "id": rule["id"],
                "priority": rule.get("priority", 0),
                "description": rule.get("description", ""),
                "when": rule.get("when", {}),
                "then": rule.get("then", []),
            }
            for rule in rules
        ]
        valid_rule_ids = [str(rule["id"]) for rule in compact_rules]
        valid_agent_names = [
            str(agent.get("name") or agent.get("id") or "").strip()
            for agent in agents
            if str(agent.get("name") or agent.get("id") or "").strip()
        ]

        system_prompt = f"""
You are the deterministic decision engine behind a WebSocket assistant.
Perform semantic functional-rule classification AND answer reasoning in one pass.

RULE MATCHING SCOPE IS STRICT:
1. The RULE MATCH TARGET is the LATEST USER MESSAGE only.
2. CONVERSATION CONTEXT is answer context, not a second source of current user intent.
3. You may use conversation context only to resolve references in the latest message, such as
   pronouns, omitted nouns, or which previously mentioned person/object the user refers to.
4. After resolving a reference, classify the operation/request expressed by the latest message
   itself. Do NOT inherit a previous turn's rule intent merely because the latest message refers
   to an entity introduced during that previous rule.
5. Example pattern: if an earlier turn requested operation X, and the latest message only asks
   when an actor involved in X is available, the X rule does NOT match unless the latest message
   itself asks for X again.
6. Evaluate each request/clause in the latest message independently.

RULE CLASSIFICATION IS FIRST AND MANDATORY:
1. Compare the latest user message independently with every semantic pre-rule.
2. A single latest user message MAY match zero, one, or multiple rules.
3. Include EVERY applicable rule in matched_rules; never discard an applicable rule merely
   because another rule has a higher priority.
4. If satisfying a request in the latest user message necessarily requires the kind of computation
   or operation described by a rule, that rule applies even when phrased as an ordinary question.
5. For each match return rule_id plus semantic confidence from 0 to 1.
6. Order matched_rules by rule priority, highest first. If none apply, return an empty array.

ANSWER COVERAGE:
- Always consider the entire LATEST USER MESSAGE, including multiple questions or requests.
- A matched rule covers only the request described by that rule; it does not make unrelated
  requests in the same latest message disappear.
- If one or more rules match and other requests remain, answer those other requests too.
- When a matching response rule permits reformulation, produce a natural answer for the entire
  latest user message while preserving the rule's required outcome and without inventing facts.
- If information needed for an uncovered request is unavailable, say so or suggest an appropriate
  available read-only agent instead of silently repeating the matched rule response.

RULE ACTIONS ARE EXECUTED LOCALLY BY FASTAPI:
- Do not try to execute rule references, nested then/catch branches, or agents yourself.
- The backend validates every proposed match and executes all accepted rules.
- Multiple rule `respond` outputs are composed locally into one assistant message in priority order.
- You may still produce an answer for normal reasoning or for reformulatable response rules.

WHEN NO RULE MATCHES:
- Answer from the supplied conversation context when possible.
- If reliable information is insufficient, use not_found or insufficient_information.
- You may suggest at most one available code-defined agent, but never claim it executed.

SEMANTIC PRE-RULES:
{json.dumps(compact_rules, ensure_ascii=False)}

AVAILABLE CODE AGENTS:
{json.dumps(agents, ensure_ascii=False)}
""".strip()

        user_prompt = f"""
RULE MATCH TARGET — LATEST USER MESSAGE:
{user_message}

ANSWER CONTEXT — DO NOT INHERIT PRIOR RULE INTENT FROM THIS SECTION:
{rendered_context or '(none)'}
""".strip()

        result = await self._complete_assistant_decision(
            system_prompt,
            user_prompt,
            valid_rule_ids,
            valid_agent_names,
            thread_id=thread_id,
            emit=emit,
        )
        payload = self._parse_assistant_decision(result.text, valid_rule_ids)
        payload["thread_id"] = result.thread_id
        await emit_flow(
            emit,
            "rules.pre.decision_parsed",
            rule_id=payload.get("matched_rule"),
            rule_ids=[match["rule_id"] for match in payload.get("matched_rules", [])],
            match_count=len(payload.get("matched_rules", [])),
            confidence=payload.get("rule_confidence"),
            parse_mode=payload.get("parse_mode"),
            rule_match_scope="latest_user_message",
            context_usage="coreference_only_no_intent_inheritance",
            thinking_mode="disabled",
            thinking_control="native_think_false" if self.decision_client else "uncontrolled",
            prompt_layout="system_user",
            structured_output="json_schema" if self.decision_client else "prompt_only",
            temperature=0 if self.decision_client else None,
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
    def _normalize_matched_rules(
        cls,
        parsed: dict[str, Any],
        valid_rule_ids: list[str],
    ) -> tuple[list[dict[str, Any]], str]:
        valid = set(valid_rule_ids)
        raw_matches = parsed.get("matched_rules")
        parse_mode = "json"

        if not isinstance(raw_matches, list):
            legacy_rule_id = parsed.get("matched_rule", parsed.get("rule_id"))
            if legacy_rule_id is None:
                raw_matches = []
            else:
                raw_matches = [
                    {
                        "rule_id": legacy_rule_id,
                        "confidence": parsed.get(
                            "rule_confidence",
                            parsed.get("confidence"),
                        ),
                    }
                ]
                parse_mode = "json_legacy_single_rule"

        matches: list[dict[str, Any]] = []
        seen: set[str] = set()
        rejected = False
        for raw_match in raw_matches:
            if isinstance(raw_match, str):
                rule_id = raw_match.strip()
                confidence = None
            elif isinstance(raw_match, dict):
                rule_id = str(
                    raw_match.get("rule_id", raw_match.get("id", "")) or ""
                ).strip()
                confidence = cls._normalize_confidence(
                    raw_match.get("confidence", raw_match.get("rule_confidence"))
                )
            else:
                rejected = True
                continue

            if not rule_id or rule_id not in valid or rule_id in seen:
                rejected = True
                continue
            seen.add(rule_id)
            matches.append({"rule_id": rule_id, "confidence": confidence})

        if rejected and parse_mode == "json":
            parse_mode = "json_with_rejected_matches"
        return matches, parse_mode

    @classmethod
    def _parse_assistant_decision(cls, text: str, valid_rule_ids: list[str]) -> dict:
        raw = (text or "").strip()
        parsed = cls._json_object(raw, {})

        if parsed:
            matches, parse_mode = cls._normalize_matched_rules(parsed, valid_rule_ids)
            primary = matches[0] if matches else None

            status = parsed.get("status")
            if status not in {"answered", "not_found", "insufficient_information"}:
                status = "answered" if parsed.get("answer") or matches else "insufficient_information"

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
                "matched_rules": matches,
                # Compatibility fields for existing UI/clients. The array above is canonical.
                "matched_rule": primary["rule_id"] if primary else None,
                "rule_confidence": primary["confidence"] if primary else None,
                "status": status,
                "answer": answer,
                "suggested_agent": suggested_agent,
                "suggested_agent_args": suggested_args,
                "parse_mode": parse_mode,
            }

        rule_decision = cls._parse_rule_decision(raw, valid_rule_ids)
        matched_rule = rule_decision.get("rule_id")
        confidence = rule_decision.get("confidence")
        matches = (
            [{"rule_id": matched_rule, "confidence": confidence}]
            if matched_rule
            else []
        )
        return {
            "matched_rules": matches,
            "matched_rule": matched_rule,
            "rule_confidence": confidence,
            "status": "answered" if matched_rule else "insufficient_information",
            "answer": None,
            "suggested_agent": None,
            "suggested_agent_args": {},
            "parse_mode": rule_decision.get("parse_mode"),
        }

    @classmethod
    def _parse_rule_decision(cls, text: str, valid_rule_ids: list[str]) -> dict:
        """Parse an explicit rule-only decision used as a tolerant legacy fallback."""
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
