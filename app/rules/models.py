from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FunctionalRule(BaseModel):
    id: str
    enabled: bool = True
    phase: Literal["pre", "post"]
    priority: int = 0
    description: str = ""
    when: dict[str, Any] = Field(default_factory=dict)
    then: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("then", mode="before")
    @classmethod
    def normalize_then(cls, value: Any) -> Any:
        """Use an ordered action list while accepting legacy single-action rule files."""
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value


class RuleFile(BaseModel):
    version: int = 1
    rules: list[FunctionalRule] = Field(default_factory=list)
