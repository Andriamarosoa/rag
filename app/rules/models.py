from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FunctionalRule(BaseModel):
    id: str
    enabled: bool = True
    phase: Literal["pre", "post"]
    priority: int = 0
    description: str = ""
    when: dict[str, Any] = Field(default_factory=dict)
    then: dict[str, Any] = Field(default_factory=dict)


class RuleFile(BaseModel):
    version: int = 1
    rules: list[FunctionalRule] = Field(default_factory=list)
