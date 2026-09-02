from __future__ import annotations

import json
import re

from app.context.manager import Summarizer
from app.sessions.models import ChatMessage

from .client import CodexMcpClient, CodexResult


class CodexService(Summarizer):
    def __init__(self, client: CodexMcpClient):
        self.client = client

    async def complete(self, prompt: str, thread_id: str | None = None) -> CodexResult:
        return await self.client.ask(prompt, thread_id=thread_id)

    async def summarize_context(self, previous_summary: str, messages: list[ChatMessage]) -> str:
        transcript = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
        prompt = f"""
You are compacting a long-running application chat context.
Preserve facts, user preferences stated in the chat, decisions, unresolved questions,
important identifiers, tool outcomes, and constraints. Remove repetition and chit-chat.
Do not invent facts. Return only the new rolling summary in plain text.

PREVIOUS SUMMARY:
{previous_summary or '(none)'}

MESSAGES TO COMPACT:
{transcript}
""".strip()
        result = await self.complete(prompt)
        return result.text.strip()

    async def choose_pre_rule(self, user_message: str, rendered_context: str, rules: list[dict]) -> dict:
        compact_rules = [
            {
                "id": r["id"],
                "description": r.get("description", ""),
                "when": r.get("when", {}),
                "then": r.get("then", {}),
            }
            for r in rules
        ]
        prompt = f"""
Select the single functional rule that clearly applies to the user's latest message.
Rules describe meanings, not literal phrases. If no rule clearly applies, return null.
Return STRICT JSON only: {{"rule_id": string|null, "confidence": number}}.

RULES:
{json.dumps(compact_rules, ensure_ascii=False)}

CONTEXT:
{rendered_context}

LATEST USER MESSAGE:
{user_message}
""".strip()
        result = await self.complete(prompt)
        return self._json_object(result.text, {"rule_id": None, "confidence": 0.0})

    async def reformulate(self, canonical_answer: str, user_message: str, rendered_context: str) -> str:
        prompt = f"""
Rephrase the canonical answer naturally for the current user message.
Do not add any fact not contained in the canonical answer.
Return only the final answer.

CANONICAL ANSWER:
{canonical_answer}

USER MESSAGE:
{user_message}

CONTEXT:
{rendered_context}
""".strip()
        return (await self.complete(prompt)).text.strip()

    async def answer(self, user_message: str, rendered_context: str, agents: list[dict], thread_id: str | None) -> dict:
        prompt = f"""
You are the local reasoning engine behind a WebSocket assistant.
Answer using the available context. If the context is insufficient, do not invent an answer.
You may suggest one of the code-defined agents, but DO NOT claim it has executed.

Return STRICT JSON only with this shape:
{{
  "status": "answered" | "not_found" | "insufficient_information",
  "answer": string | null,
  "suggested_agent": string | null,
  "suggested_agent_args": object
}}

AVAILABLE CODE AGENTS:
{json.dumps(agents, ensure_ascii=False)}

CONTEXT:
{rendered_context}

LATEST USER MESSAGE:
{user_message}
""".strip()
        result = await self.complete(prompt, thread_id=thread_id)
        payload = self._json_object(
            result.text,
            {
                "status": "insufficient_information",
                "answer": None,
                "suggested_agent": None,
                "suggested_agent_args": {},
            },
        )
        payload["thread_id"] = result.thread_id
        return payload

    @staticmethod
    def _json_object(text: str, fallback: dict) -> dict:
        text = text.strip()
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else fallback
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else fallback
            except Exception:
                pass
        return fallback
