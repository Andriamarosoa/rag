from app.codex.service import CodexService


VALID = ["password_reset"]


def test_rule_decision_parses_strict_json():
    result = CodexService._parse_rule_decision(
        '{"rule_id":"password_reset","confidence":0.97}',
        VALID,
    )
    assert result == {
        "rule_id": "password_reset",
        "confidence": 0.97,
        "parse_mode": "json",
    }


def test_rule_decision_parses_json_surrounded_by_model_noise():
    result = CodexService._parse_rule_decision(
        'Decision:\n```json\n{"rule_id":"password_reset","confidence":0.91}\n```',
        VALID,
    )
    assert result["rule_id"] == "password_reset"
    assert result["confidence"] == 0.91
    assert result["parse_mode"] == "json"


def test_rule_decision_parses_key_value_fallback():
    result = CodexService._parse_rule_decision(
        "rule_id=password_reset confidence=94",
        VALID,
    )
    assert result == {
        "rule_id": "password_reset",
        "confidence": 0.94,
        "parse_mode": "key_value",
    }


def test_rule_decision_accepts_exact_rule_id_without_fake_zero_confidence():
    result = CodexService._parse_rule_decision("password_reset", VALID)
    assert result == {
        "rule_id": "password_reset",
        "confidence": None,
        "parse_mode": "exact_rule_id",
    }


def test_rule_decision_does_not_match_arbitrary_rule_id_occurrence():
    result = CodexService._parse_rule_decision(
        "I saw the password_reset rule in the prompt but I will not provide a decision.",
        VALID,
    )
    assert result == {
        "rule_id": None,
        "confidence": None,
        "parse_mode": "unparseable",
    }


def test_rule_decision_rejects_unknown_rule_id():
    result = CodexService._parse_rule_decision(
        '{"rule_id":"made_up_rule","confidence":1}',
        VALID,
    )
    assert result["rule_id"] is None
    assert result["parse_mode"] == "unparseable"
