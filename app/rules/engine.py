from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.codex.service import CodexService
from app.flow.events import FlowEmitter, emit_flow

from .models import FunctionalRule, RuleFile


class RuleEngine:
    def __init__(self, rules_path: Path, codex: CodexService):
        self.rules_path = rules_path
        self.codex = codex
        self._rules: RuleFile | None = None

    def reload(self) -> RuleFile:
        payload = json.loads(self.rules_path.read_text(encoding="utf-8"))
        self._rules = RuleFile.model_validate(payload)
        return self._rules

    @property
    def rules(self) -> RuleFile:
        return self._rules or self.reload()

    async def match_pre(
        self,
        user_message: str,
        rendered_context: str,
        emit: FlowEmitter | None = None,
    ) -> FunctionalRule | None:
        rules = sorted(
            [r for r in self.rules.rules if r.enabled and r.phase == "pre"],
            key=lambda r: r.priority,
            reverse=True,
        )
        semantic = [r for r in rules if r.when.get("type") == "semantic"]
        await emit_flow(
            emit,
            "rules.pre.started",
            candidate_count=len(semantic),
        )
        if not semantic:
            await emit_flow(emit, "rules.pre.no_match", reason="no_semantic_rules")
            return None

        decision = await self.codex.choose_pre_rule(
            user_message=user_message,
            rendered_context=rendered_context,
            rules=[r.model_dump() for r in semantic],
            emit=emit,
        )
        rule_id = decision.get("rule_id")
        confidence = float(decision.get("confidence") or 0)
        if not rule_id or confidence < 0.65:
            await emit_flow(
                emit,
                "rules.pre.no_match",
                reason="below_confidence_threshold" if rule_id else "model_returned_null",
                proposed_rule_id=rule_id,
                confidence=confidence,
                threshold=0.65,
            )
            return None

        matched = next((r for r in semantic if r.id == rule_id), None)
        if matched is None:
            await emit_flow(
                emit,
                "rules.pre.no_match",
                reason="unknown_rule_id",
                proposed_rule_id=rule_id,
                confidence=confidence,
            )
            return None

        await emit_flow(
            emit,
            "rules.pre.matched",
            rule_id=matched.id,
            confidence=confidence,
            priority=matched.priority,
            action_type=matched.then.get("type"),
        )
        return matched

    def match_post(self, result: dict[str, Any]) -> list[FunctionalRule]:
        matched: list[FunctionalRule] = []
        rules = sorted(
            [r for r in self.rules.rules if r.enabled and r.phase == "post"],
            key=lambda r: r.priority,
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
