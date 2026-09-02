from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ClientEnvelope(BaseModel):
    type: Literal["chat.message", "agent.execute", "rules.reload", "ping"]
    request_id: str | None = None
    user_id: str
    chat_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ServerEnvelope(BaseModel):
    type: str
    request_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
