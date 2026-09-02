from __future__ import annotations

from typing import Any

from app.flow.events import FlowEmitter

from .segmented_service import SegmentedDecisionService


class StrictSegmentedDecisionService(SegmentedDecisionService):
    """Final guard: actionable non-rule segments need an explicit response body."""

    async def answer_with_rules(
        self,
        user_message: str,
        rendered_context: str,
        rules: list[dict],
        agents: list[dict],
        thread_id: str | None,
        emit: FlowEmitter | None = None,
    ) -> dict:
        result = await super().answer_with_rules(
            user_message=user_message,
            rendered_context=rendered_context,
            rules=rules,
            agents=agents,
            thread_id=thread_id,
            emit=emit,
        )

        segments = result.get("segments")
        if not isinstance(segments, list):
            return result

        unresolved = [
            str(value).strip()
            for value in result.get("unresolved_requests", [])
            if str(value).strip()
        ]
        unresolved_keys = {value.casefold() for value in unresolved}
        supplemental = str(result.get("supplemental_answer") or "").strip()
        supplemental_parts = [supplemental] if supplemental else []

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            kind = str(segment.get("kind") or "")
            matched_rule_ids = segment.get("matched_rule_ids")
            if not isinstance(matched_rule_ids, list):
                matched_rule_ids = []
            status = str(segment.get("status") or "")
            response = str(segment.get("response") or "").strip()

            if (
                kind in {"question", "request"}
                and not matched_rule_ids
                and status == "answered"
                and not response
            ):
                text = str(segment.get("text") or "").strip()
                response = f"I do not have enough reliable information to answer: {text}"
                segment["status"] = "unanswered"
                segment["response"] = response
                if text and text.casefold() not in unresolved_keys:
                    unresolved.append(text)
                    unresolved_keys.add(text.casefold())
                supplemental_parts.append(response)

        result["unresolved_requests"] = unresolved
        if unresolved:
            result["has_unanswered_requests"] = True

        deduped_parts: list[str] = []
        seen: set[str] = set()
        for part in supplemental_parts:
            normalized = " ".join(part.split()).casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped_parts.append(part)
        result["supplemental_answer"] = " ".join(deduped_parts).strip() or None
        return result
