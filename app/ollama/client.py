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

    Codex currently talks to Ollama through the OpenAI-compatible Responses API. The native
    `/api/chat` endpoint is used for the integrated assistant decision because it exposes the
    real `think=false` control and native timing/token metrics.
    """

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        think: bool = False,
    ) -> OllamaNativeResult:
        response = await self._client.post(
            "/api/chat",
            json={
                "model": model or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "think": think,
                "stream": False,
                "format": "json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message") or {}
        return OllamaNativeResult(
            text=str(message.get("content") or "").strip(),
            raw=payload,
        )
