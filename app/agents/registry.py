from __future__ import annotations

from typing import Any

from .base import AgentResult, CodeAgent


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, CodeAgent] = {}

    def register(self, agent: CodeAgent) -> None:
        if agent.spec.name in self._agents:
            raise ValueError(f"agent_already_registered:{agent.spec.name}")
        self._agents[agent.spec.name] = agent

    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": agent.spec.name,
                "description": agent.spec.description,
                "input_schema": agent.spec.input_schema,
                "write_action": agent.spec.write_action,
                "requires_confirmation": agent.spec.requires_confirmation,
            }
            for agent in self._agents.values()
        ]

    def get(self, name: str) -> CodeAgent | None:
        return self._agents.get(name)

    async def execute(self, name: str, arguments: dict[str, Any], confirmed: bool = False) -> AgentResult:
        agent = self.get(name)
        if not agent:
            return AgentResult(ok=False, error="agent_not_found")
        if agent.spec.requires_confirmation and not confirmed:
            return AgentResult(ok=False, error="confirmation_required", data={"agent": name, "arguments": arguments})
        return await agent.execute(arguments)
