from pathlib import Path

from app.rules.engine import RuleEngine


def test_partial_answer_triggers_email_fallback():
    engine = RuleEngine(Path("config/rules"))
    engine.reload()

    matched = engine.match_post(
        {
            "status": "answered",
            "has_unanswered_requests": True,
            "unresolved_requests": ["When is the administrator available?"],
        }
    )

    assert "no_answer_suggest_email" in [rule.id for rule in matched]


def test_fully_answered_message_does_not_trigger_email_fallback():
    engine = RuleEngine(Path("config/rules"))
    engine.reload()

    matched = engine.match_post(
        {
            "status": "answered",
            "has_unanswered_requests": False,
            "unresolved_requests": [],
        }
    )

    assert "no_answer_suggest_email" not in [rule.id for rule in matched]
