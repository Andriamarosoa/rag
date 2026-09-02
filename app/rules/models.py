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

    @field_validator("then", mode="before")
    @classmethod
    def normalize_then(cls, value: Any) -> Any:
        """Canonicalize top-level `then` to a list while accepting object/string shorthand."""
        if value is None:
            return []
        if isinstance(value, (dict, str)):
            return [value]
        return value


class RuleFile(BaseModel):
    version: int = 1
    rules: list[FunctionalRule] = Field(default_factory=list)
