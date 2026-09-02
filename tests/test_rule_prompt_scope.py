import asyncio
import json

from app.codex.client import CodexResult
from app.codex.service import CodexService
from app.ollama.client import OllamaNativeResult


class FakeCodexClient:
    model = "qwen3:8b"

    async def ask(self, prompt: str, thread_id: str | None = None) -> CodexResult:
        return CodexResult(text="{}", thread_id=thread_id)


class CapturingDecisionClient:
    model = "qwen3:8b"

    def __init__(self):
        self.system_prompt = ""
        self.user_prompt = ""

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
        self.system_prompt = system_prompt or ""
        self.user_prompt = user_prompt or ""
        return OllamaNativeResult(
            text=json.dumps(
                {
                    "matched_rules": [],
                    "status": "insufficient_information",
                    "answer": None,
                    "suggested_agent": None,
                    "suggested_agent_args": {},
                }
            ),
            raw={},
        )


def test_rule_prompt_does_not_inherit_previous_turn_intent():
    decision = CapturingDecisionClient()
    service = CodexService(FakeCodexClient(), decision_client=decision)

    asyncio.run(
        service.answer_with_rules(
            user_message="When is the administrator available?",
            rendered_context=(
                "USER: I want to reset my password\n"
                "ASSISTANT: Contact your administrator to reset the password."
            ),
            rules=[
                {
                    "id": "password_reset",
                    "priority": 100,
                    "description": "Reset a forgotten password",
                    "when": {
                        "type": "semantic",
                        "scope": "latest_user_message",
                        "context_usage": "coreference_only_no_intent_inheritance",
                    },
                    "then": [
                        {
                            "type": "respond",
                            "canonical_answer": "Contact your administrator.",
                        }
                    ],
                }
            ],
            agents=[],
            thread_id=None,
        )
    )

    assert "RULE MATCHING SCOPE IS STRICT" in decision.system_prompt
    assert "Do NOT inherit a previous turn's rule intent" in decision.system_prompt
    assert "RULE MATCH TARGET — LATEST USER MESSAGE:" in decision.user_prompt
    assert "When is the administrator available?" in decision.user_prompt
    assert "ANSWER CONTEXT — DO NOT INHERIT PRIOR RULE INTENT" in decision.user_prompt


def test_prompt_requires_uncovered_subquestions_to_be_answered():
    decision = CapturingDecisionClient()
    service = CodexService(FakeCodexClient(), decision_client=decision)

    asyncio.run(
        service.answer_with_rules(
            user_message="Reset my password. When is the administrator available?",
            rendered_context="",
            rules=[],
            agents=[],
            thread_id=None,
        )
    )

    assert "A matched rule covers only the request described by that rule" in decision.system_prompt
    assert "If one or more rules match and other requests remain, answer those other requests too" in decision.system_prompt
