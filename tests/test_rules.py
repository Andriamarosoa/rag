import json
from pathlib import Path

from app.rules.engine import RuleEngine
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


def test_then_accepts_rule_id_references_mixed_with_actions():
    rule = FunctionalRule.model_validate(
        {
            "id": "composed",
            "phase": "pre",
            "when": {"type": "semantic"},
            "then": [
                "shared_response",
                {"type": "suggest_agent", "agent": "send_email"},
            ],
        }
    )

    assert rule.then == [
        "shared_response",
        {"type": "suggest_agent", "agent": "send_email"},
    ]


def test_single_rule_id_then_is_normalized_to_array():
    rule = FunctionalRule.model_validate(
        {
            "id": "composed",
            "phase": "pre",
            "when": {"type": "semantic"},
            "then": "shared_response",
        }
    )

    assert rule.then == ["shared_response"]


def test_ref_object_accepts_action_overrides():
    rule = FunctionalRule.model_validate(
        {
            "id": "composed",
            "phase": "pre",
            "when": {"type": "semantic"},
            "then": {
                "ref": "shared_email",
                "label": "Send an email too",
                "requires_confirmation": False,
            },
        }
    )

    assert rule.then == [
        {
            "ref": "shared_email",
            "label": "Send an email too",
            "requires_confirmation": False,
        }
    ]
    assert RuleEngine.action_labels(rule) == ["rule:shared_email"]


def test_action_can_have_recursive_then_and_catch_branches():
    node = {
        "type": "respond",
        "canonical_answer": "ok",
        "then": {
            "ref": "shared_email",
            "label": "Email support",
        },
        "catch": [
            "fallback_rule",
            {
                "type": "respond",
                "canonical_answer": "fallback",
            },
        ],
    }
    rule = FunctionalRule.model_validate(
        {
            "id": "recursive",
            "phase": "pre",
            "when": {"type": "semantic"},
            "then": node,
        }
    )

    assert rule.then == [node]


def test_control_group_without_type_is_allowed_as_then_object():
    group = {
        "then": "primary_rule",
        "catch": {
            "type": "respond",
            "canonical_answer": "fallback",
        },
    }
    rule = FunctionalRule.model_validate(
        {
            "id": "grouped",
            "phase": "pre",
            "when": {"type": "semantic"},
            "then": group,
        }
    )

    assert rule.then == [group]
    assert RuleEngine.action_labels(rule) == ["group"]
