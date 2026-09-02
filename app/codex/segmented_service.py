from __future__ import annotations

import json
import re
from typing import Any

from app.flow.events import FlowEmitter, emit_flow

from .isolated_service import IsolatedDecisionService


class SegmentedDecisionService(IsolatedDecisionService):
    """Decision pipeline that makes every latest-message segment explicit.

    Segmentation happens locally before any model call. The rule classifier receives only
    those numbered segments and no conversation history. Answer reasoning must report a
    result for every segment. Missing actionable segment results are treated as unanswered
    by the backend instead of being silently dropped.
    """

    @staticmethod
    def _segment_latest_message(text: str) -> list[dict[str, str]]:
        raw = str(text or "").strip()
        if not raw:
            return []
        pieces = [
            piece.strip()
            for piece in re.split(r"(?<=[.!?])\s+|\n+|(?<=;)\s*", raw)
            if piece and piece.strip()
        ]
        if not pieces:
            pieces = [raw]
        return [
            {"segment_id": f"s{index}", "text": piece}
            for index, piece in enumerate(pieces, start=1)
        ]

    @staticmethod
    def _default_segment_kind(text: str) -> str:
        return "question" if "?" in str(text or "") else "statement"

    @staticmethod
    def _segment_classifier_schema(
        valid_rule_ids: list[str],
        segment_ids: list[str],
    ) -> dict[str, Any]:
        match_schema = {
            "type": "object",
            "properties": {
                "rule_id": {"type": "string", "enum": valid_rule_ids},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["rule_id", "confidence"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "segment_id": {"type": "string", "enum": segment_ids},
                            "kind": {
                                "type": "string",
                                "enum": ["question", "request", "statement"],
                            },
                            "matched_rules": {
                                "type": "array",
                                "items": match_schema,
                            },
                        },
                        "required": ["segment_id", "kind", "matched_rules"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["segments"],
            "additionalProperties": False,
        }

    @classmethod
    def _segment_answer_schema(
        cls,
        valid_agent_names: list[str],
        segment_ids: list[str],
    ) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["answered", "not_found", "insufficient_information"],
                },
                "answer": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                },
                "segment_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "segment_id": {"type": "string", "enum": segment_ids},
                            "status": {
                                "type": "string",
                                "enum": [
                                    "answered",
                                    "unanswered",
                                    "acknowledged",
                                    "covered_by_rule",
                                ],
                            },
                            "response": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                            },
                        },
                        "required": ["segment_id", "status", "response"],
                        "additionalProperties": False,
                    },
                },
                "suggested_agent": cls._nullable_enum(valid_agent_names),
                "suggested_agent_args": {"type": "object"},
            },
            "required": [
                "status",
                "answer",
                "segment_results",
                "suggested_agent",
                "suggested_agent_args",
            ],
            "additionalProperties": False,
        }

    @classmethod
    def _normalize_segment_matches(
        cls,
        raw: Any,
        valid_rule_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        valid = set(valid_rule_ids)
        matches: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("rule_id") or "").strip()
            confidence = cls._normalize_confidence(item.get("confidence"))
            if not rule_id or rule_id not in valid or rule_id in seen or confidence is None:
                continue
            seen.add(rule_id)
            matches.append({"rule_id": rule_id, "confidence": confidence})
        return matches

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
        priority_by_id = {rule["id"]: rule["priority"] for rule in compact_rules}
        rules_by_id = {str(rule["id"]): rule for rule in rules}
        valid_agent_names = [
            str(agent.get("name") or agent.get("id") or "").strip()
            for agent in agents
            if str(agent.get("name") or agent.get("id") or "").strip()
        ]

        segments = self._segment_latest_message(user_message)
        segment_ids = [item["segment_id"] for item in segments]
        segment_text_by_id = {item["segment_id"]: item["text"] for item in segments}

        classifier_system = f"""
You are a semantic functional-rule classifier.
You receive PRE-SEGMENTED pieces of the latest user message and NO conversation history.

MANDATORY:
- Return one classification object for EVERY supplied segment id. Never omit a segment.
- Preserve segment independence. Do not merge segments.
- kind=question for a question, request for an instruction/desired operation, statement otherwise.
- Compare each segment independently with every semantic rule.
- Similar vocabulary is not sufficient; the segment must actually request what the rule describes.
- Return every applicable rule for that segment with confidence 0..1.

SEMANTIC RULES:
{json.dumps(compact_rules, ensure_ascii=False)}
""".strip()
        classifier_user = "LATEST USER SEGMENTS ONLY:\n" + json.dumps(
            segments,
            ensure_ascii=False,
        )

        classifier_result = await self._native_json_operation(
            operation="rule_classifier",
            system_prompt=classifier_system,
            user_prompt=classifier_user,
            schema=self._segment_classifier_schema(valid_rule_ids, segment_ids),
            thread_id=thread_id,
            emit=emit,
        )
        classifier_payload = self._json_object(classifier_result.text, {})
        raw_classifications = classifier_payload.get("segments")
        if not isinstance(raw_classifications, list):
            raw_classifications = []
        classification_by_id = {
            str(item.get("segment_id") or ""): item
            for item in raw_classifications
            if isinstance(item, dict) and str(item.get("segment_id") or "") in segment_text_by_id
        }

        tracked_segments: list[dict[str, Any]] = []
        aggregated_by_rule: dict[str, float] = {}
        for segment in segments:
            segment_id = segment["segment_id"]
            classified = classification_by_id.get(segment_id, {})
            kind = str(classified.get("kind") or "").strip()
            if kind not in {"question", "request", "statement"}:
                kind = self._default_segment_kind(segment["text"])
            segment_matches = self._normalize_segment_matches(
                classified.get("matched_rules"),
                valid_rule_ids,
            )
            for match in segment_matches:
                previous = aggregated_by_rule.get(match["rule_id"])
                if previous is None or match["confidence"] > previous:
                    aggregated_by_rule[match["rule_id"]] = match["confidence"]
            tracked_segments.append(
                {
                    "segment_id": segment_id,
                    "text": segment["text"],
                    "kind": kind,
                    "matched_rules": segment_matches,
                }
            )

        matches = [
            {"rule_id": rule_id, "confidence": confidence}
            for rule_id, confidence in aggregated_by_rule.items()
        ]
        matches.sort(
            key=lambda item: (priority_by_id.get(item["rule_id"], 0), item["confidence"]),
            reverse=True,
        )

        actionable_uncovered = [
            segment
            for segment in tracked_segments
            if segment["kind"] in {"question", "request"} and not segment["matched_rules"]
        ]
        has_uncovered_request = bool(actionable_uncovered)
        has_nonrule_segment = any(not segment["matched_rules"] for segment in tracked_segments)

        await emit_flow(
            emit,
            "rules.pre.decision_parsed",
            rule_id=matches[0]["rule_id"] if matches else None,
            rule_ids=[match["rule_id"] for match in matches],
            match_count=len(matches),
            confidence=matches[0]["confidence"] if matches else None,
            parse_mode="segmented_json",
            classifier_context_isolated=True,
            classifier_context_tokens=0,
            segment_count=len(tracked_segments),
            has_uncovered_request=has_uncovered_request,
            uncovered_segment_ids=[item["segment_id"] for item in actionable_uncovered],
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
        needs_answer_reasoning = (
            not matches or has_nonrule_segment or needs_reformulation
        )

        segment_results: dict[str, dict[str, Any]] = {}
        status = "answered" if matches else "insufficient_information"
        model_answer: str | None = None
        suggested_agent: str | None = None
        suggested_args: dict[str, Any] = {}

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
You are the answer-reasoning stage. Rule matching is already frozen.
You MUST NOT add, remove, or change rule matches.

TRACKED USER SEGMENTS:
{json.dumps(tracked_segments, ensure_ascii=False)}

FROZEN MATCHED RULES:
{json.dumps(selected_rules, ensure_ascii=False)}

MANDATORY COVERAGE CONTRACT:
- Return exactly one segment_result for EVERY tracked segment id.
- Never silently drop a segment.
- For a segment handled by a frozen rule use covered_by_rule, unless you also provide its natural response.
- For a question/request you can reliably answer, use answered and provide response text.
- For a question/request you cannot reliably answer, use unanswered and provide a concise explanation.
- For a non-actionable statement, use acknowledged and optionally provide a natural acknowledgement.
- `answer` should be a natural combined response covering all segments in their original order.
- Do not invent facts.
- You may suggest at most one available agent when useful; never claim it executed.

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
                schema=self._segment_answer_schema(valid_agent_names, segment_ids),
                thread_id=thread_id,
                emit=emit,
            )
            parsed_answer = self._json_object(answer_result.text, {})
            parsed_status = parsed_answer.get("status")
            if parsed_status in {"answered", "not_found", "insufficient_information"}:
                status = parsed_status
            raw_answer = parsed_answer.get("answer")
            if raw_answer is not None:
                model_answer = str(raw_answer).strip() or None
            raw_segment_results = parsed_answer.get("segment_results")
            if isinstance(raw_segment_results, list):
                for item in raw_segment_results:
                    if not isinstance(item, dict):
                        continue
                    segment_id = str(item.get("segment_id") or "")
                    if segment_id not in segment_text_by_id or segment_id in segment_results:
                        continue
                    item_status = str(item.get("status") or "")
                    if item_status not in {
                        "answered",
                        "unanswered",
                        "acknowledged",
                        "covered_by_rule",
                    }:
                        continue
                    response = item.get("response")
                    response_text = str(response).strip() if response is not None else ""
                    segment_results[segment_id] = {
                        "status": item_status,
                        "response": response_text or None,
                    }
            candidate_agent = parsed_answer.get("suggested_agent")
            if candidate_agent in valid_agent_names:
                suggested_agent = candidate_agent
            candidate_args = parsed_answer.get("suggested_agent_args")
            if isinstance(candidate_args, dict):
                suggested_args = candidate_args

        normalized_results: list[dict[str, Any]] = []
        unresolved_requests: list[str] = []
        supplemental_parts: list[str] = []
        for segment in tracked_segments:
            segment_id = segment["segment_id"]
            item = segment_results.get(segment_id)
            if item is None:
                if segment["matched_rules"]:
                    item = {"status": "covered_by_rule", "response": None}
                elif segment["kind"] in {"question", "request"}:
                    item = {
                        "status": "unanswered",
                        "response": f"I do not have enough reliable information to answer: {segment['text']}",
                    }
                else:
                    item = {"status": "acknowledged", "response": None}

            item_status = item["status"]
            response = item.get("response")
            normalized_results.append(
                {
                    "segment_id": segment_id,
                    "text": segment["text"],
                    "kind": segment["kind"],
                    "matched_rule_ids": [
                        match["rule_id"] for match in segment["matched_rules"]
                    ],
                    "status": item_status,
                    "response": response,
                }
            )

            if item_status == "unanswered" and segment["kind"] in {"question", "request"}:
                unresolved_requests.append(segment["text"])
            if not segment["matched_rules"] and response:
                supplemental_parts.append(str(response).strip())

        has_unanswered_requests = bool(unresolved_requests) or status in {
            "not_found",
            "insufficient_information",
        }
        if model_answer is None:
            all_responses = [
                str(item.get("response") or "").strip()
                for item in normalized_results
                if str(item.get("response") or "").strip()
            ]
            model_answer = " ".join(all_responses) or None

        supplemental_answer = " ".join(
            part for part in supplemental_parts if part
        ).strip() or None

        primary = matches[0] if matches else None
        return {
            "matched_rules": matches,
            "matched_rule": primary["rule_id"] if primary else None,
            "rule_confidence": primary["confidence"] if primary else None,
            "status": status,
            "answer": model_answer,
            "segments": normalized_results,
            "unresolved_requests": unresolved_requests,
            "has_unanswered_requests": has_unanswered_requests,
            "supplemental_answer": supplemental_answer,
            "suggested_agent": suggested_agent,
            "suggested_agent_args": suggested_args,
            "parse_mode": "segmented_json",
            "has_uncovered_request": has_uncovered_request,
            "decision_pipeline": "segmented_isolated_classifier_then_answer",
            "thread_id": thread_id,
        }
