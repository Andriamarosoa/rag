import pytest

from app.agents.registry import AgentRegistry
from app.agents.base import AgentResult, AgentSpec, CodeAgent


class FakeWriteAgent(CodeAgent):
    spec = AgentSpec(
        name="write",
        description="test",
        input_schema={"type": "object"},
        write_action=True,
        requires_confirmation=True,
    )

    async def execute(self, arguments):
        return AgentResult(ok=True, data=arguments)


@pytest.mark.asyncio
async def test_write_agent_requires_confirmation():
    registry = AgentRegistry()
    registry.register(FakeWriteAgent())
    denied = await registry.execute("write", {"x": 1}, confirmed=False)
    assert denied.ok is False
    assert denied.error == "confirmation_required"

    allowed = await registry.execute("write", {"x": 1}, confirmed=True)
    assert allowed.ok is True
