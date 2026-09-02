from app.agents.base import AgentResult, AgentSpec, CodeAgent
from app.agents.registry import AgentRegistry
from app.rules.models import FunctionalRule, RuleFile
from app.segmented_orchestrator import SegmentedOrchestrator


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


async def test_suggest_agent_template_is_passed_to_ui_without_becoming_answer_text():
    template = "<p>Vous pouver envoyer un email via this <link>{{labe}}</link></p>"
    rule = FunctionalRule.model_validate(
        {
            "id": "no_answer_suggest_email",
            "phase": "post",
            "when": {
                "type": "result_state",
                "field": "has_unanswered_requests",
                "operator": "eq",
                "value": True,
            },
            "then": [
                {
                    "type": "suggest_agent",
                    "agent": "send_email",
                    "label": "Send an email",
                    "template": template,
                    "requires_confirmation": True,
                }
            ],
        }
    )

    agents = AgentRegistry()
    agents.register(DummyEmailAgent())
    orchestrator = SegmentedOrchestrator(
        store=None,  # type: ignore[arg-type]
        context=None,  # type: ignore[arg-type]
        rules=DummyRuleEngine([rule]),  # type: ignore[arg-type]
        agents=agents,
        codex=None,  # type: ignore[arg-type]
    )
    result = {"status": "answered", "answer": "Main answer", "actions": []}

    success = await orchestrator._apply_rule_actions(
        rule,
        result,
        None,
        source="post_rule",
        allow_model_reformulation=False,
    )

    assert success is True
    assert result["answer"] == "Main answer"
    assert result["actions"] == [
        {
            "type": "suggest_agent",
            "agent": "send_email",
            "label": "Send an email",
            "arguments": {},
            "requires_confirmation": True,
            "rule_id": "no_answer_suggest_email",
            "origin_rule_id": "no_answer_suggest_email",
            "template": template,
        }
    ]
    assert "post-message" not in result["actions"][0]
    assert "_post_messages" not in result
