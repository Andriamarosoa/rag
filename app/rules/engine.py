from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.flow.events import FlowEmitter, emit_flow

from .models import FunctionalRule, RuleFile


class RuleEngine:
    """Loads business rules and enforces model decisions locally.

    Semantic classification is intentionally performed by the same model pass that
    produces the assistant answer. This engine never calls a model by itself.
    """

    def __init__(self, rules_path: Path):
        self.rules_path = rules_path
        self._rules: RuleFile | None = None

    def reload(self) -> RuleFile:
        payload = json.loads(self.rules_path.read_text(encoding="utf-8"))
        self._rules = RuleFile.model_validate(payload)
        return self._rules

    @property
    def rules(self) -> RuleFile:
        return self._rules or self.reload()

    def semantic_pre_rules(self) -> list[FunctionalRule]:
        return sorted(
            [
                rule
                for rule in self.rules.rules
                if rule.enabled
                and rule.phase == "pre"
                and rule.when.get("type") == "semantic"
            ],
            key=lambda rule: rule.priority,
            reverse=True,
        )

    async def resolve_pre_decision(
        self,
        decision: dict[str, Any],
        emit: FlowEmitter | None = None,
        decision_elapsed_ms: float | None = None,
    ) -> FunctionalRule | None:
        semantic = self.semantic_pre_rules()
        rule_id = decision.get("matched_rule") or decision.get("rule_id")
        raw_confidence = decision.get("rule_confidence")
        if raw_confidence is None:
            raw_confidence = decision.get("confidence")
        confidence = float(raw_confidence) if raw_confidence is not None else None
        parse_mode = decision.get("parse_mode")

        timing: dict[str, Any] = {
            "timing_scope": "integrated_assistant_decision",
        }
        if decision_elapsed_ms is not None:
            timing.update(
                {
                    "decision_elapsed_ms": decision_elapsed_ms,
                    "decision_elapsed_seconds": round(decision_elapsed_ms / 1000, 3),
                }
            )

        if not semantic:
            await emit_flow(
                emit,
                "rules.pre.no_match",
                reason="no_semantic_rules",
                **timing,
            )
            return None

        if not rule_id:
            await emit_flow(
                emit,
                "rules.pre.no_match",
                reason=(
                    "unparseable_model_output"
                    if parse_mode == "unparseable"
                    else "model_returned_null"
                ),
                proposed_rule_id=None,
                confidence=confidence,
                threshold=0.65,
                parse_mode=parse_mode,
                **timing,
            )
            return None

        if confidence is not None and confidence < 0.65:
            await emit_flow(
                emit,
                "rules.pre.no_match",
                reason="below_confidence_threshold",
                proposed_rule_id=rule_id,
                confidence=confidence,
                threshold=0.65,
                parse_mode=parse_mode,
                **timing,
            )
            return None

        matched = next((rule for rule in semantic if rule.id == rule_id), None)
        if matched is None:
            await emit_flow(
                emit,
                "rules.pre.no_match",
                reason="unknown_rule_id",
                proposed_rule_id=rule_id,
                confidence=confidence,
                parse_mode=parse_mode,
                **timing,
            )
            return None

        await emit_flow(
            emit,
            "rules.pre.matched",
            rule_id=matched.id,
            confidence=confidence,
            priority=matched.priority,
            action_type=matched.then.get("type"),
            parse_mode=parse_mode,
            **timing,
        )
        return matched

    def match_post(self, result: dict[str, Any]) -> list[FunctionalRule]:
        matched: list[FunctionalRule] = []
        rules = sorted(
            [rule for rule in self.rules.rules if rule.enabled and rule.phase == "post"],
            key=lambda rule: rule.priority,
            reverse=True,
        )
        for rule in rules:
            condition = rule.when
            if condition.get("type") != "result_state":
                continue
            field = condition.get("field")
            operator = condition.get("operator")
            expected = condition.get("value")
            actual = result.get(field)
            if operator == "in" and isinstance(expected, list) and actual in expected:
                matched.append(rule)
            elif operator == "eq" and actual == expected:
                matched.append(rule)
        return matched
