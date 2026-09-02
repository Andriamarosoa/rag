from app.rules.models import FunctionalRule


def test_semantic_rule_defaults_to_latest_user_message_scope():
    rule = FunctionalRule.model_validate(
        {
            "id": "password_reset",
            "phase": "pre",
            "when": {"type": "semantic"},
            "then": {"type": "respond", "canonical_answer": "reset"},
        }
    )

    assert rule.when["scope"] == "latest_user_message"
    assert rule.when["context_usage"] == "coreference_only_no_intent_inheritance"


def test_respond_action_keeps_uncovered_latest_message_requests_in_scope():
    rule = FunctionalRule.model_validate(
        {
            "id": "password_reset",
            "phase": "pre",
            "when": {"type": "semantic"},
            "then": {
                "type": "respond",
                "canonical_answer": "reset",
                "then": {
                    "type": "respond",
                    "canonical_answer": "nested",
                },
            },
        }
    )

    first = rule.then[0]
    nested = first["then"]

    assert first["answer_scope"] == "full_latest_user_message"
    assert (
        first["uncovered_request_policy"]
        == "also_answer_other_requests_in_the_latest_user_message_not_covered_by_this_rule"
    )
    assert nested["answer_scope"] == "full_latest_user_message"
