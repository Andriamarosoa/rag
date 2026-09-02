from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# A rule node can be a concrete/control object or a short rule-id reference string.
# Objects may recursively contain `then` and `catch`, and may reference another rule with `ref`.
RuleThenItem = dict[str, Any] | str


class FunctionalRule(BaseModel):
    id: str
    enabled: bool = True
    phase: Literal["pre", "post"]
    priority: int = 0
    description: str = ""
    when: dict[str, Any] = Field(default_factory=dict)
    then: list[RuleThenItem] = Field(default_factory=list)

    @field_validator("when", mode="before")
    @classmethod
    def normalize_when(cls, value: Any) -> Any:
        """Give semantic rules a safe conversational matching scope by default.

        Semantic matching targets the latest user turn. Conversation history may resolve
        references such as "it" or "the administrator", but must not carry a previous
        rule intent forward into a different follow-up question.
        """
        if value is None:
            return {}
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if str(normalized.get("type") or "").strip() == "semantic":
            normalized.setdefault("scope", "latest_user_message")
            normalized.setdefault("context_usage", "coreference_only_no_intent_inheritance")
        return normalized

    @classmethod
    def _annotate_response_policy(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._annotate_response_policy(item) for item in value]
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        if str(normalized.get("type") or "").strip() == "respond":
            normalized.setdefault("answer_scope", "full_latest_user_message")
            normalized.setdefault(
                "uncovered_request_policy",
                "also_answer_other_requests_in_the_latest_user_message_not_covered_by_this_rule",
            )

        if "then" in normalized:
            normalized["then"] = cls._annotate_response_policy(normalized["then"])
        if "catch" in normalized:
            normalized["catch"] = cls._annotate_response_policy(normalized["catch"])
        return normalized

    @field_validator("then", mode="before")
    @classmethod
    def normalize_then(cls, value: Any) -> Any:
        """Canonicalize top-level `then` and annotate response coverage semantics."""
        if value is None:
            return []
        if isinstance(value, (dict, str)):
            value = [value]
        return cls._annotate_response_policy(value)


class RuleFile(BaseModel):
    version: int = 1
    rules: list[FunctionalRule] = Field(default_factory=list)
