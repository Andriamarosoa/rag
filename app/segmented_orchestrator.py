from __future__ import annotations

from typing import Any

from app.flow.events import FlowEmitter, emit_flow
from app.orchestrator import Orchestrator
from app.rules.models import FunctionalRule


class SegmentedOrchestrator(Orchestrator):
    """Orchestrator variant that preserves answers for non-rule message segments."""

    @staticmethod
    def _normalized_text(value: Any) -> str:
        return " ".join(str(value or "").split()).casefold()

    @staticmethod
    def _compose_rule_outputs(result: dict[str, Any]) -> list[dict[str, Any]]:
        outputs = Orchestrator._compose_rule_outputs(result)
        supplemental = str(result.get("supplemental_answer") or "").strip()
        if not supplemental:
            return outputs

        current = str(result.get("answer") or "").strip()
        normalized_current = SegmentedOrchestrator._normalized_text(current)
        normalized_supplemental = SegmentedOrchestrator._normalized_text(supplemental)

        if normalized_supplemental and normalized_supplemental not in normalized_current:
            result["answer"] = (
                f"{current}\n\n{supplemental}" if current else supplemental
            )
            if result.get("answer"):
                result["status"] = "answered"

        return outputs

    async def _execute_concrete_action(
        self,
        action: dict[str, Any],
        rule: FunctionalRule,
        result: dict[str, Any],
        emit: FlowEmitter | None,
        *,
        source: str,
        origin_rule_id: str,
        action_index: int,
        action_count: int,
        control_depth: int,
        allow_model_reformulation: bool,
    ) -> bool:
        """Keep UI templates attached to suggested-agent actions.

        `template` is a presentation-only field. It is sent to the browser unchanged and
        never appended to the assistant answer. The browser owns the pseudo `<link>`
        replacement and binds it to the same action handler as the normal action button.
        """
        action_type = str(action.get("type") or "").strip()
        if action_type != "suggest_agent":
            return await super()._execute_concrete_action(
                action,
                rule,
                result,
                emit,
                source=source,
                origin_rule_id=origin_rule_id,
                action_index=action_index,
                action_count=action_count,
                control_depth=control_depth,
                allow_model_reformulation=allow_model_reformulation,
            )

        agent_name = str(action.get("agent") or "").strip()
        agent = self.agents.get(agent_name) if agent_name else None
        if agent is None:
            await emit_flow(
                emit,
                "rule.action.failed",
                rule_id=rule.id,
                origin_rule_id=origin_rule_id,
                action_type=action_type,
                action_index=action_index,
                action_count=action_count,
                reason="unknown_agent",
                agent=agent_name or None,
            )
            return False

        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        requires_confirmation = bool(
            action.get("requires_confirmation", agent.spec.requires_confirmation)
        )
        label = str(
            action.get("label") or agent_name.replace("_", " ").title()
        ).strip()
        template = str(action.get("template") or "").strip()

        ui_action: dict[str, Any] = {
            "type": "suggest_agent",
            "agent": agent_name,
            "label": label,
            "arguments": arguments,
            "requires_confirmation": requires_confirmation,
            "rule_id": rule.id,
            "origin_rule_id": origin_rule_id,
        }
        if template:
            ui_action["template"] = template

        result.setdefault("actions", []).append(ui_action)
        await emit_flow(
            emit,
            "agent.suggested",
            agent=agent_name,
            source=source,
            rule_id=rule.id,
            origin_rule_id=origin_rule_id,
            action_index=action_index,
            requires_confirmation=requires_confirmation,
            arguments=arguments,
            template=template or None,
        )
        await emit_flow(
            emit,
            "rule.action.completed",
            rule_id=rule.id,
            origin_rule_id=origin_rule_id,
            action_type="suggest_agent",
            action_index=action_index,
            action_count=action_count,
            status="suggested",
            template=template or None,
        )
        return True
