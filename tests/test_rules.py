import json
from pathlib import Path

import pytest

from app.rules.engine import RuleEngine
from app.rules.models import FunctionalRule


def test_rule_directory_is_valid_and_filename_is_rule_id():
    rules = RuleEngine(Path("config/rules")).reload()

    assert rules.version == 1
    assert {rule.id for rule in rules.rules} == {
        "math_calculation",
        "password_reset",
        "no_answer_suggest_email",
    }
    assert all(isinstance(rule.then, list) for rule in rules.rules)
    assert all(rule.then for rule in rules.rules)

    for path in Path("config/rules").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "id" not in payload
        assert any(rule.id == path.stem for rule in rules.rules)


def test_loader_injects_id_from_filename(tmp_path: Path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "my_rule.json").write_text(
        json.dumps(
            {
                "phase": "pre",
                "when": {"type": "semantic"},
                "then": {"type": "respond", "canonical_answer": "ok"},
            }
        ),
        encoding="utf-8",
    )

    rules = RuleEngine(rules_dir).reload()

    assert len(rules.rules) == 1
    assert rules.rules[0].id == "my_rule"
    assert rules.rules[0].then == [{"type": "respond", "canonical_answer": "ok"}]


def test_loader_rejects_explicit_id_in_rule_file(tmp_path: Path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "filename_id.json").write_text(
        json.dumps(
            {
                "id": "other_id",
                "phase": "pre",
                "when": {"type": "semantic"},
                "then": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rule_id_must_come_from_filename"):
        RuleEngine(rules_dir).reload()


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
