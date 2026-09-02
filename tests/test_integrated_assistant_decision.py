import asyncio
import json

from app.codex.client import CodexResult
from app.codex.service import CodexService
from app.ollama.client import OllamaNativeResult


class FakeCodexClient:
    def __init__(self):
        self.model = "qwen3:8b"
        self.calls = 0

    async def ask(self, prompt: str, thread_id: str | None = None) -> CodexResult:
        self.calls += 1
        return CodexResult(text="{}", thread_id=thread_id)


class FakeDecisionClient:
    def __init__(self, response: dict):
        self.model = "qwen3:8b"
        self.response = response
        self.calls = 0
        self.last_prompt = ""
        self.last_think: bool | None = None

    async def chat_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        think: bool = False,
    ) -> OllamaNativeResult:
        self.calls += 1
        self.last_prompt = prompt
        self.last_think = think
        return OllamaNativeResult(
            text=json.dumps(self.response),
            raw={
                "total_duration": 1_500_000_000,
                "load_duration": 100_000_000,
                "prompt_eval_count": 700,
                "prompt_eval_duration": 700_000_000,
                "eval_count": 42,
                "eval_duration": 600_000_000,
            },
        )


def test_rules_and_reasoning_use_one_native_no_think_model_call():
    codex_client = FakeCodexClient()
    decision_client = FakeDecisionClient(
        {
            "matched_rule": "math_calculation",
            "rule_confidence": 0.99,
            "status": "answered",
            "answer": "hahahaha",
            "suggested_agent": None,
            "suggested_agent_args": {},
        }
    )
    service = CodexService(codex_client, decision_client=decision_client)
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

    assert decision_client.calls == 1
    assert codex_client.calls == 0
    assert decision_client.last_think is False
    assert "/no_think" not in decision_client.last_prompt
    assert result["matched_rule"] == "math_calculation"
    assert result["rule_confidence"] == 0.99
    assert result["answer"] == "hahahaha"

    started = next(data for event, data in events if event == "model.request.started")
    completed = next(data for event, data in events if event == "model.request.completed")

    assert started["provider"] == "ollama_native"
    assert started["operation"] == "assistant_decision"
    assert started["thinking_mode"] == "disabled"
    assert started["thinking_control"] == "native_think_false"
    assert started["timing_scope"] == "ollama_native_api"

    assert completed["provider"] == "ollama_native"
    assert completed["thinking_mode"] == "disabled"
    assert completed["native_ollama_eval_metrics_available"] is True
    assert completed["metrics_source"] == "ollama_native_response"
    assert completed["ollama_total_seconds"] == 1.5
    assert completed["ollama_prompt_eval_count"] == 700
    assert completed["ollama_eval_count"] == 42
    assert completed["ollama_prompt_tokens_per_second"] == 1000.0
    assert completed["ollama_output_tokens_per_second"] == 70.0


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
