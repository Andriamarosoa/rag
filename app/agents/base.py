from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    write_action: bool = False
    requires_confirmation: bool = False


@dataclass(slots=True)
class AgentResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class CodeAgent(ABC):
    spec: AgentSpec

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> AgentResult:
        raise NotImplementedError
