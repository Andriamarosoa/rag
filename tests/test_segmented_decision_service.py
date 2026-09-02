import asyncio
import json

from app.codex.client import CodexResult
from app.codex.segmented_service import SegmentedDecisionService
from app.ollama.client import OllamaNativeResult
from app.segmented_orchestrator import SegmentedOrchestrator


class FakeCodexClient:
    model = "qwen3:8b"

    async def ask(self, prompt: str, thread_id: str | None = None) -> CodexResult:
        return CodexResult(text="{}", thread_id=thread_id)


class SequenceDecisionClient:
    def __init__(self, responses: list[dict]):
        self.model = "qwen3:8b"
        self.responses = list(responses)

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
        return OllamaNativeResult(
            text=json.dumps(self.responses.pop(0)),
            raw={
                "total_duration": 500_000_000,
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
        "description": "The user asks to reset a password.",
        "when": {"type": "semantic"},
        "then": [
            {
                "type": "respond",
                "canonical_answer": "Contact the administrator.",
                "reformulate": True,
            }
        ],
    }


def test_local_segmentation_keeps_every_sentence():
    segments = SegmentedDecisionService._segment_latest_message(
        "I want to reset password. When is the administrator available?\nI love banana."
    )

    assert segments == [
        {"segment_id": "s1", "text": "I want to reset password."},
        {"segment_id": "s2", "text": "When is the administrator available?"},
        {"segment_id": "s3", "text": "I love banana."},
    ]


def test_missing_question_result_is_forced_to_unanswered():
    decision = SequenceDecisionClient(
        [
            {
                "segments": [
                    {
                        "segment_id": "s1",
                        "kind": "request",
                        "matched_rules": [
                            {"rule_id": "password_reset", "confidence": 0.98}
                        ],
                    },
                    {
                        "segment_id": "s2",
                        "kind": "question",
                        "matched_rules": [],
                    },
                    {
                        "segment_id": "s3",
                        "kind": "statement",
                        "matched_rules": [],
                    },
                ]
            },
            {
                "status": "answered",
                "answer": "Contact the administrator. I love banana.",
                # Deliberately omit s2 to reproduce the model dropping the availability question.
                "segment_results": [
                    {
                        "segment_id": "s1",
                        "status": "covered_by_rule",
                        "response": "Contact the administrator.",
                    },
                    {
                        "segment_id": "s3",
                        "status": "acknowledged",
                        "response": "I love banana.",
                    },
                ],
                "suggested_agent": None,
                "suggested_agent_args": {},
            },
        ]
    )
    service = SegmentedDecisionService(FakeCodexClient(), decision_client=decision)

    result = asyncio.run(
        service.answer_with_rules(
            user_message=(
                "I want to reset password. When is the administrator available?\n"
                "I love banana."
            ),
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
    assert result["unresolved_requests"] == [
        "When is the administrator available?"
    ]
    assert result["has_unanswered_requests"] is True
    s2 = next(item for item in result["segments"] if item["segment_id"] == "s2")
    assert s2["status"] == "unanswered"
    assert "When is the administrator available?" in s2["response"]
    assert "When is the administrator available?" in result["supplemental_answer"]


def test_segmented_composer_preserves_supplemental_answer():
    result = {
        "answer": "model answer",
        "_rule_outputs": [
            {
                "rule_id": "password_reset",
                "origin_rule_id": "password_reset",
                "source": "pre_rule",
                "content": "Contact the administrator.",
            }
        ],
        "supplemental_answer": "Administrator availability is not known.",
    }

    SegmentedOrchestrator._compose_rule_outputs(result)

    assert result["answer"] == (
        "Contact the administrator.\n\nAdministrator availability is not known."
    )
