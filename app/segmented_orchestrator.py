from __future__ import annotations

from typing import Any

from app.orchestrator import Orchestrator


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
