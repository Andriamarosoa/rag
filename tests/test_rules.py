import json
from pathlib import Path

from app.rules.models import FunctionalRule, RuleFile


def test_rule_file_is_valid_and_uses_action_arrays():
    payload = json.loads(Path("config/rules.json").read_text(encoding="utf-8"))
    rules = RuleFile.model_validate(payload)

    assert rules.version == 1
    assert any(rule.id == "math_calculation" for rule in rules.rules)
    assert any(rule.id == "password_reset" for rule in rules.rules)
    assert any(rule.id == "no_answer_suggest_email" for rule in rules.rules)
    assert all(isinstance(rule.then, list) for rule in rules.rules)
    assert all(rule.then for rule in rules.rules)


def test_legacy_single_then_action_is_normalized_to_array():
    rule = FunctionalRule.model_validate(
        {
            "id": "legacy",
            "phase": "pre",
            "when": {"type": "semantic"},
            "then": {"type": "respond", "canonical_answer": "ok"},
        }
    )

    assert rule.then == [{"type": "respond", "canonical_answer": "ok"}]
