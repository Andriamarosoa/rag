import asyncio
import json

from app.codex.client import CodexResult
from app.codex.service import CodexService


class FakeClient:
    def __init__(self, response: dict):
        self.model = "qwen3:8b"
        self.response = response
        self.calls = 0

    async def ask(self, prompt: str, thread_id: str | None = None) -> CodexResult:
        self.calls += 1
        return CodexResult(
            text=json.dumps(self.response),
            thread_id=thread_id or "thread-1",
        )


def test_rules_and_reasoning_share_one_model_call():
    client = FakeClient(
        {
            "matched_rule": "math_calculation",
            "rule_confidence": 0.99,
            "status": "answered",
            "answer": "hahahaha",
            "suggested_agent": None,
            "suggested_agent_args": {},
        }
    )
    service = CodexService(client)
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, data: dict) -> None:
        events.append((event_type, data))

    result = asyncio.run(
        service.answer_with_rules(
            user_message="2 + 2",
            rendered_context="",
            rules=[
                {
                    "id": "math_calculation",
                    "priority": 200,
                    "description": "Any mathematical calculation.",
                    "when": {"type": "semantic"},
                    "then": {
                        "type": "respond",
                        "canonical_answer": "hahahaha",
                        "reformulate": False,
                    },
                }
            ],
            agents=[],
            thread_id=None,
            emit=emit,
        )
    )

    assert client.calls == 1
    assert result["matched_rule"] == "math_calculation"
    assert result["rule_confidence"] == 0.99
    assert result["answer"] == "hahahaha"
    assert result["thread_id"] == "thread-1"

    started = next(data for event, data in events if event == "model.request.started")
    completed = next(data for event, data in events if event == "model.request.completed")

    assert started["operation"] == "assistant_decision"
    assert started["model"] == "qwen3:8b"
    assert started["prompt_tokens_estimated"] > 0
    assert started["timing_scope"] == "codex_to_ollama_round_trip"

    assert completed["elapsed_ms"] >= 0
    assert completed["prompt_tokens_estimated"] == started["prompt_tokens_estimated"]
    assert completed["output_tokens_estimated"] > 0
    assert completed["metrics_source"] == "application_wall_clock"
    assert completed["native_ollama_eval_metrics_available"] is False


def test_combined_parser_keeps_normal_answer_without_rule():
    result = CodexService._parse_assistant_decision(
        json.dumps(
            {
                "matched_rule": None,
                "rule_confidence": 0.93,
                "status": "answered",
                "answer": "A normal answer",
                "suggested_agent": None,
                "suggested_agent_args": {},
            }
        ),
        ["math_calculation", "password_reset"],
    )

    assert result["matched_rule"] is None
    assert result["status"] == "answered"
    assert result["answer"] == "A normal answer"
    assert result["parse_mode"] == "json"
