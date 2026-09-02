import asyncio
from pathlib import Path

from app.rules.engine import RuleEngine
from app.rules.models import FunctionalRule, RuleFile


def make_engine() -> RuleEngine:
    engine = RuleEngine(Path("unused"))
    engine._rules = RuleFile(
        rules=[
            FunctionalRule.model_validate(
                {
                    "id": "low_priority",
                    "phase": "pre",
                    "priority": 100,
                    "when": {"type": "semantic"},
                    "then": {"type": "respond", "canonical_answer": "low"},
                }
            ),
            FunctionalRule.model_validate(
                {
                    "id": "high_priority",
                    "phase": "pre",
                    "priority": 200,
                    "when": {"type": "semantic"},
                    "then": {"type": "respond", "canonical_answer": "high"},
                }
            ),
        ]
    )
    return engine


def test_multiple_matches_are_all_accepted_and_sorted_by_priority():
    engine = make_engine()
    decision = {
        "matched_rules": [
            {"rule_id": "low_priority", "confidence": 0.95},
            {"rule_id": "high_priority", "confidence": 0.90},
        ],
        "parse_mode": "json",
    }

    matches = asyncio.run(engine.resolve_pre_decisions(decision))

    assert [rule.id for rule in matches] == ["high_priority", "low_priority"]


def test_invalid_match_is_rejected_without_discarding_other_matches():
    engine = make_engine()
    decision = {
        "matched_rules": [
            {"rule_id": "high_priority", "confidence": 0.97},
            {"rule_id": "low_priority", "confidence": 0.40},
            {"rule_id": "missing", "confidence": 0.99},
        ],
        "parse_mode": "json",
    }

    matches = asyncio.run(engine.resolve_pre_decisions(decision))

    assert [rule.id for rule in matches] == ["high_priority"]


def test_legacy_single_match_still_resolves():
    engine = make_engine()
    decision = {
        "matched_rule": "low_priority",
        "rule_confidence": 0.92,
        "parse_mode": "json_legacy_single_rule",
    }

    matches = asyncio.run(engine.resolve_pre_decisions(decision))

    assert [rule.id for rule in matches] == ["low_priority"]
