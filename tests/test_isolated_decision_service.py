import asyncio
import json

from app.codex.client import CodexResult
from app.codex.isolated_service import IsolatedDecisionService
from app.ollama.client import OllamaNativeResult


class FakeCodexClient:
    model = "qwen3:8b"

    async def ask(self, prompt: str, thread_id: str | None = None) -> CodexResult:
        return CodexResult(text="{}", thread_id=thread_id)


class SequenceDecisionClient:
    def __init__(self, responses: list[dict]):
        self.model = "qwen3:8b"
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def chat_json(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        format_schema: dict | None = None,
        model: str | None = None,
        think: bool = False,
        temperature: float = 0.0,
    ) -> OllamaNativeResult:
        self.calls.append(
            {
                "system_prompt": system_prompt or "",
                "user_prompt": user_prompt if user_prompt is not None else (prompt or ""),
                "schema": format_schema or {},
                "think": think,
                "temperature": temperature,
            }
        )
        response = self.responses.pop(0)
        return OllamaNativeResult(
            text=json.dumps(response),
            raw={
                "total_duration": 500_000_000,
                "load_duration": 10_000_000,
                "prompt_eval_count": 100,
                "prompt_eval_duration": 200_000_000,
                "eval_count": 20,
                "eval_duration": 250_000_000,
            },
        )


def password_rule() -> dict:
    return {
        "id": "password_reset",
        "priority": 100,
        "description": (
            "The current user message explicitly asks to recover, reset, change, or regain "
            "access to a forgotten password."
        ),
        "when": {"type": "semantic", "scope": "latest_user_message"},
        "then": [
            {
                "type": "respond",
                "canonical_answer": "Contact the administrator.",
                "reformulate": True,
            }
        ],
    }


def test_followup_classifier_never_receives_previous_reset_context():
    history = (
        "USER: I want to reset password.\n"
        "ASSISTANT: The user must contact their administrator to reset the password."
    )
    decision = SequenceDecisionClient(
        [
            {
                "matched_rules": [],
                "has_uncovered_request": True,
            },
            {
                "status": "insufficient_information",
                "answer": "I cannot determine the administrator's availability.",
                "suggested_agent": None,
                "suggested_agent_args": {},
            },
        ]
    )
    service = IsolatedDecisionService(FakeCodexClient(), decision_client=decision)

    result = asyncio.run(
        service.answer_with_rules(
            user_message="When is the administrator available?",
            rendered_context=history,
            rules=[password_rule()],
            agents=[],
            thread_id=None,
            emit=None,
        )
    )

    assert result["matched_rules"] == []
    assert result["matched_rule"] is None
    assert result["answer"] == "I cannot determine the administrator's availability."
    assert result["decision_pipeline"] == "isolated_classifier_then_answer"
    assert len(decision.calls) == 2

    classifier_call = decision.calls[0]
    answer_call = decision.calls[1]

    assert "When is the administrator available?" in classifier_call["user_prompt"]
    assert "I want to reset password" not in classifier_call["system_prompt"]
    assert "I want to reset password" not in classifier_call["user_prompt"]
    assert "CONVERSATION CONTEXT" not in classifier_call["user_prompt"]

    assert "I want to reset password" in answer_call["user_prompt"]
    assert "matched_rules" in classifier_call["schema"]["properties"]
    assert "matched_rules" not in answer_call["schema"]["properties"]


def test_mixed_message_keeps_rule_match_and_answers_uncovered_question():
    decision = SequenceDecisionClient(
        [
            {
                "matched_rules": [
                    {"rule_id": "password_reset", "confidence": 0.98},
                ],
                "has_uncovered_request": True,
            },
            {
                "status": "answered",
                "answer": (
                    "The user must contact their administrator to reset the password. "
                    "I cannot determine when the administrator is available."
                ),
                "suggested_agent": None,
                "suggested_agent_args": {},
            },
        ]
    )
    service = IsolatedDecisionService(FakeCodexClient(), decision_client=decision)

    result = asyncio.run(
        service.answer_with_rules(
            user_message="I want to reset password. When is the administrator available?",
            rendered_context="",
            rules=[password_rule()],
            agents=[],
            thread_id=None,
            emit=None,
        )
    )

    assert result["matched_rules"] == [
        {"rule_id": "password_reset", "confidence": 0.98}
    ]
    assert result["has_uncovered_request"] is True
    assert "administrator to reset" in result["answer"]
    assert "available" in result["answer"]
    assert len(decision.calls) == 2
