from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.flow.events import FlowEmitter, emit_flow

from .models import FunctionalRule, RuleFile


_RULE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class RuleEngine:
    """Loads business rules and enforces model decisions locally.

    Each `*.json` file under `rules_path` defines one rule. The filename stem is the rule id,
    so `config/rules/password_reset.json` becomes rule id `password_reset`. Rule JSON files
    must not contain an `id` field.

    Semantic classification is intentionally performed by the same model pass that
    produces the assistant answer. This engine never calls a model by itself.
    """

    def __init__(self, rules_path: Path):
        self.rules_path = rules_path
        self._rules: RuleFile | None = None

    def reload(self) -> RuleFile:
        if not self.rules_path.exists():
            raise FileNotFoundError(f"rules_path_not_found:{self.rules_path}")
        if not self.rules_path.is_dir():
            raise ValueError(f"rules_path_must_be_directory:{self.rules_path}")

        files = sorted(self.rules_path.glob("*.json"), key=lambda path: path.name.casefold())
        rules: list[FunctionalRule] = []
        seen_ids: set[str] = set()

        for path in files:
            rule_id = path.stem
            if not rule_id or not _RULE_ID_RE.fullmatch(rule_id):
                raise ValueError(f"invalid_rule_filename:{path.name}")

            normalized_id = rule_id.casefold()
            if normalized_id in seen_ids:
                raise ValueError(f"duplicate_rule_id:{rule_id}")
            seen_ids.add(normalized_id)

            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"rule_file_must_be_object:{path.name}")
            if "id" in payload:
                raise ValueError(f"rule_id_must_come_from_filename:{path.name}")

            rules.append(FunctionalRule.model_validate({"id": rule_id, **payload}))

        self._rules = RuleFile(version=1, rules=rules)
        return self._rules

    @property
    def rules(self) -> RuleFile:
        return self._rules or self.reload()

    def get_rule(self, rule_id: str, *, include_disabled: bool = False) -> FunctionalRule | None:
        rule = next((candidate for candidate in self.rules.rules if candidate.id == rule_id), None)
        if rule is None:
            return None
        if not include_disabled and not rule.enabled:
            return None
        return rule

    @staticmethod
    def action_labels(rule: FunctionalRule) -> list[str]:
        labels: list[str] = []
        for item in rule.then:
            if isinstance(item, str):
                labels.append(f"rule:{item}")
                continue
            if not isinstance(item, dict):
                labels.append("invalid")
                continue
            if item.get("ref"):
                labels.append(f"rule:{item['ref']}")
            elif item.get("type"):
                labels.append(str(item["type"]))
            elif "then" in item:
                labels.append("group")
            else:
                labels.append("unknown")
        return labels

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

    @staticmethod
    def _decision_matches(decision: dict[str, Any]) -> list[dict[str, Any]]:
        matches = decision.get("matched_rules")
        if isinstance(matches, list):
            return [item for item in matches if isinstance(item, dict)]

        # Temporary compatibility with the former singular decision contract.
        rule_id = decision.get("matched_rule") or decision.get("rule_id")
        if not rule_id:
            return []
        confidence = decision.get("rule_confidence")
        if confidence is None:
            confidence = decision.get("confidence")
        return [{"rule_id": rule_id, "confidence": confidence}]

    async def resolve_pre_decisions(
        self,
        decision: dict[str, Any],
        emit: FlowEmitter | None = None,
        decision_elapsed_ms: float | None = None,
    ) -> list[FunctionalRule]:
        semantic = self.semantic_pre_rules()
        parse_mode = decision.get("parse_mode")
        threshold = 0.65

        timing: dict[str, Any] = {
            "timing_scope": "integrated_assistant_decision",
        }
        if decision_elapsed_ms is not None:
            decision_elapsed_seconds = round(decision_elapsed_ms / 1000, 3)
            timing.update(
                {
                    "decision_elapsed_ms": decision_elapsed_ms,
                    "decision_elapsed_seconds": decision_elapsed_seconds,
                    "status": f"durée décision: {decision_elapsed_seconds:.1f} s",
                }
            )

        if not semantic:
            await emit_flow(
                emit,
                "rules.pre.no_match",
                reason="no_semantic_rules",
                match_count=0,
                **timing,
            )
            return []

        proposed = self._decision_matches(decision)
        if not proposed:
            await emit_flow(
                emit,
                "rules.pre.no_match",
                reason=(
                    "unparseable_model_output"
                    if parse_mode == "unparseable"
                    else "model_returned_empty_matches"
                ),
                proposed_rule_ids=[],
                threshold=threshold,
                parse_mode=parse_mode,
                match_count=0,
                **timing,
            )
            return []

        semantic_by_id = {rule.id: rule for rule in semantic}
        accepted: list[FunctionalRule] = []
        accepted_ids: set[str] = set()

        for proposed_index, match in enumerate(proposed):
            rule_id = str(match.get("rule_id") or "").strip()
            raw_confidence = match.get("confidence")
            try:
                confidence = float(raw_confidence) if raw_confidence is not None else None
            except (TypeError, ValueError):
                confidence = None

            reason: str | None = None
            matched = semantic_by_id.get(rule_id)
            if not rule_id:
                reason = "missing_rule_id"
            elif rule_id in accepted_ids:
                reason = "duplicate_rule_id"
            elif confidence is not None and confidence < threshold:
                reason = "below_confidence_threshold"
            elif matched is None:
                reason = "unknown_or_nonsemantic_rule_id"

            if reason:
                await emit_flow(
                    emit,
                    "rules.pre.rejected",
                    proposed_rule_id=rule_id or None,
                    proposed_index=proposed_index,
                    confidence=confidence,
                    threshold=threshold,
                    reason=reason,
                    parse_mode=parse_mode,
                )
                continue

            accepted_ids.add(rule_id)
            accepted.append(matched)

        accepted.sort(key=lambda rule: rule.priority, reverse=True)

        if not accepted:
            await emit_flow(
                emit,
                "rules.pre.no_match",
                reason="all_proposed_rules_rejected",
                proposed_rule_ids=[str(item.get("rule_id") or "") for item in proposed],
                threshold=threshold,
                parse_mode=parse_mode,
                match_count=0,
                **timing,
            )
            return []

        confidence_by_id = {
            str(item.get("rule_id") or ""): item.get("confidence")
            for item in proposed
            if isinstance(item, dict)
        }
        accepted_rule_ids = [rule.id for rule in accepted]

        for accepted_index, matched in enumerate(accepted):
            action_types = self.action_labels(matched)
            await emit_flow(
                emit,
                "rules.pre.matched",
                rule_id=matched.id,
                accepted_index=accepted_index,
                accepted_count=len(accepted),
                accepted_rule_ids=accepted_rule_ids,
                confidence=confidence_by_id.get(matched.id),
                priority=matched.priority,
                action_type=action_types[0] if action_types else None,
                action_types=action_types,
                action_count=len(matched.then),
                parse_mode=parse_mode,
                **timing,
            )

        return accepted

    async def resolve_pre_decision(
        self,
        decision: dict[str, Any],
        emit: FlowEmitter | None = None,
        decision_elapsed_ms: float | None = None,
    ) -> FunctionalRule | None:
        """Compatibility wrapper returning only the highest-priority accepted rule."""
        matches = await self.resolve_pre_decisions(
            decision,
            emit=emit,
            decision_elapsed_ms=decision_elapsed_ms,
        )
        return matches[0] if matches else None

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
