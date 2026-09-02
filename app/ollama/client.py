from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class OllamaNativeResult:
    text: str
    raw: dict[str, Any]

    @property
    def total_duration_ns(self) -> int | None:
        value = self.raw.get("total_duration")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def load_duration_ns(self) -> int | None:
        value = self.raw.get("load_duration")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def prompt_eval_count(self) -> int | None:
        value = self.raw.get("prompt_eval_count")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def prompt_eval_duration_ns(self) -> int | None:
        value = self.raw.get("prompt_eval_duration")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def eval_count(self) -> int | None:
        value = self.raw.get("eval_count")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def eval_duration_ns(self) -> int | None:
        value = self.raw.get("eval_duration")
        return int(value) if isinstance(value, (int, float)) else None


class OllamaNativeClient:
    """Direct Ollama API client used when exact model controls are required.

    Codex talks to Ollama through the OpenAI-compatible API whose base URL usually ends in
    `/v1`. The native `/api/chat` API does not use that prefix, so this client deliberately
    normalizes an accidentally reused `/v1` URL before building native requests.
    """

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 180.0):
        self.base_url = self.normalize_base_url(base_url)
        self.model = model
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds)

    @staticmethod
    def normalize_base_url(base_url: str) -> str:
        normalized = base_url.strip().rstrip("/")
        if normalized.lower().endswith("/v1"):
            normalized = normalized[:-3].rstrip("/")
        return normalized

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_json(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        format_schema: dict[str, Any] | None = None,
        model: str | None = None,
        think: bool = False,
        temperature: float = 0.0,
    ) -> OllamaNativeResult:
        """Call Ollama `/api/chat` with deterministic structured output.

        `prompt` is kept for backwards compatibility. For decision routing, callers should
        prefer separate system/user messages so functional rules cannot be confused with the
        user's content.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        effective_user_prompt = user_prompt if user_prompt is not None else (prompt or "")
        messages.append({"role": "user", "content": effective_user_prompt})

        response = await self._client.post(
            "/api/chat",
            json={
                "model": model or self.model,
                "messages": messages,
                "think": think,
                "stream": False,
                "format": format_schema or "json",
                "options": {"temperature": temperature},
            },
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message") or {}
        return OllamaNativeResult(
            text=str(message.get("content") or "").strip(),
            raw=payload,
        )
