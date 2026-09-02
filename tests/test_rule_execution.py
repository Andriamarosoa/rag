from app.agents.base import AgentResult, AgentSpec, CodeAgent
from app.agents.registry import AgentRegistry
from app.orchestrator import Orchestrator
from app.rules.models import FunctionalRule, RuleFile


class DummyEmailAgent(CodeAgent):
    spec = AgentSpec(
        name="send_email",
        description="Test email agent",
        input_schema={},
        write_action=True,
        requires_confirmation=True,
    )

    async def execute(self, arguments: dict) -> AgentResult:
        return AgentResult(ok=True, data=arguments)


class DummyRuleEngine:
    def __init__(self, rules: list[FunctionalRule]):
        self._rules = RuleFile(rules=rules)

    def get_rule(self, rule_id: str, *, include_disabled: bool = False) -> FunctionalRule | None:
        rule = next((item for item in self._rules.rules if item.id == rule_id), None)
        if rule is None:
            return None
        if not include_disabled and not rule.enabled:
            return None
        return rule


def make_orchestrator(rules: list[FunctionalRule]) -> Orchestrator:
    agents = AgentRegistry()
    agents.register(DummyEmailAgent())
    return Orchestrator(
        store=None,  # type: ignore[arg-type]
        context=None,  # type: ignore[arg-type]
        rules=DummyRuleEngine(rules),  # type: ignore[arg-type]
        agents=agents,
        codex=None,  # type: ignore[arg-type]
    )


async def test_ref_object_overrides_referenced_action_without_mutating_source_rule():
    shared = FunctionalRule.model_validate(
        {
            "id": "shared_email",
            "phase": "post",
            "then": {
                "type": "suggest_agent",
                "agent": "send_email",
                "label": "Original label",
                "requires_confirmation": True,
            },
        }
    )
    root = FunctionalRule.model_validate(
        {
            "id": "root",
            "phase": "pre",
            "when": {"type": "semantic"},
            "then": {
                "ref": "shared_email",
                "label": "Overridden label",
                "requires_confirmation": False,
            },
        }
    )
    orchestrator = make_orchestrator([root, shared])
    result = {"status": "answered", "answer": "ok", "actions": []}

    success = await orchestrator._apply_rule_actions(
        root,
        result,
        None,
        source="test",
        allow_model_reformulation=False,
    )

    assert success is True
    assert result["actions"][0]["agent"] == "send_email"
    assert result["actions"][0]["label"] == "Overridden label"
    assert result["actions"][0]["requires_confirmation"] is False
    assert shared.then[0]["label"] == "Original label"
    assert shared.then[0]["requires_confirmation"] is True


async def test_nested_then_runs_after_successful_action():
    shared = FunctionalRule.model_validate(
        {
            "id": "shared_email",
            "phase": "post",
            "then": {
                "type": "suggest_agent",
                "agent": "send_email",
            },
        }
    )
    root = FunctionalRule.model_validate(
        {
            "id": "root",
            "phase": "pre",
            "when": {"type": "semantic"},
            "then": {
                "type": "respond",
                "canonical_answer": "hello",
                "then": "shared_email",
            },
        }
    )
    orchestrator = make_orchestrator([root, shared])
    result = {"status": "not_found", "answer": None, "actions": [], "_rule_outputs": []}

    success = await orchestrator._apply_rule_actions(
        root,
        result,
        None,
        source="test",
        allow_model_reformulation=False,
    )
    orchestrator._compose_rule_outputs(result)

    assert success is True
    assert result["answer"] == "hello"
    assert result["actions"][0]["agent"] == "send_email"


async def test_catch_runs_when_referenced_rule_fails():
    root = FunctionalRule.model_validate(
        {
            "id": "root",
            "phase": "pre",
            "when": {"type": "semantic"},
            "then": {
                "ref": "missing_rule",
                "catch": {
                    "type": "respond",
                    "canonical_answer": "fallback",
                },
            },
        }
    )
    orchestrator = make_orchestrator([root])
    result = {"status": "not_found", "answer": None, "actions": [], "_rule_outputs": []}

    success = await orchestrator._apply_rule_actions(
        root,
        result,
        None,
        source="test",
        allow_model_reformulation=False,
    )
    orchestrator._compose_rule_outputs(result)

    assert success is True
    assert result["answer"] == "fallback"
    assert result["matched_rule"] == "root"


async def test_multiple_rule_responses_are_composed_into_one_answer():
    high = FunctionalRule.model_validate(
        {
            "id": "high",
            "phase": "pre",
            "priority": 200,
            "when": {"type": "semantic"},
            "then": {"type": "respond", "canonical_answer": "high answer"},
        }
    )
    low = FunctionalRule.model_validate(
        {
            "id": "low",
            "phase": "pre",
            "priority": 100,
            "when": {"type": "semantic"},
            "then": [
                {"type": "respond", "canonical_answer": "low answer"},
                {"type": "suggest_agent", "agent": "send_email"},
            ],
        }
    )
    orchestrator = make_orchestrator([high, low])
    result = {
        "status": "answered",
        "answer": "model answer",
        "_model_answer": "model answer",
        "_rule_outputs": [],
        "_pre_rule_batch_count": 2,
        "actions": [],
        "matched_rule": "high",
    }

    assert await orchestrator._apply_rule_actions(
        high,
        result,
        None,
        source="pre_rule",
        allow_model_reformulation=False,
        origin_rule_id="high",
    )
    assert await orchestrator._apply_rule_actions(
        low,
        result,
        None,
        source="pre_rule",
        allow_model_reformulation=False,
        origin_rule_id="low",
    )

    outputs = orchestrator._compose_rule_outputs(result)

    assert result["answer"] == "high answer\n\nlow answer"
    assert [item["origin_rule_id"] for item in outputs] == ["high", "low"]
    assert result["matched_rule"] == "high"
    assert result["actions"][0]["agent"] == "send_email"


def test_rule_response_composer_deduplicates_identical_fragments():
    result = {
        "answer": None,
        "_rule_outputs": [
            {"rule_id": "a", "origin_rule_id": "a", "source": "pre_rule", "content": "Same text"},
            {"rule_id": "b", "origin_rule_id": "b", "source": "pre_rule", "content": " same   text "},
        ],
    }

    outputs = Orchestrator._compose_rule_outputs(result)

    assert result["answer"] == "Same text"
    assert len(outputs) == 1
