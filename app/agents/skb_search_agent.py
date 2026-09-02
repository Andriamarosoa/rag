from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.skb.client import SkbClient

from .base import AgentResult, AgentSpec, CodeAgent


class SkbSearchAgent(CodeAgent):
    _BASE_SPEC = AgentSpec(
        name="search_skb",
        description=(
            "Search the SKB knowledge base at skb.uniconsults.mu for product/module documentation. "
            "This is a read-only action and does not require confirmation."
        ),
        write_action=False,
        requires_confirmation=False,
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language question or keywords to search in SKB",
                },
                "module": {
                    "type": "string",
                    "description": "Optional SKB module name used to focus ranking",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    def __init__(self, client: SkbClient):
        self.client = client
        self.spec = deepcopy(self._BASE_SPEC)
        self._modules: list[str] = []

    @property
    def modules(self) -> list[str]:
        return list(self._modules)

    def set_modules(self, modules: list[str]) -> None:
        cleaned: list[str] = []
        seen: set[str] = set()
        for module in modules:
            value = " ".join(str(module).split()).strip()
            key = value.casefold()
            if not value or key in seen:
                continue
            seen.add(key)
            cleaned.append(value)

        self._modules = cleaned
        module_schema = self.spec.input_schema["properties"]["module"]
        if cleaned:
            module_schema["enum"] = cleaned
            module_schema["description"] = (
                "Optional SKB module name used to focus ranking. "
                f"Available modules discovered from SKB: {', '.join(cleaned)}"
            )
            self.spec.description = (
                self._BASE_SPEC.description
                + " Available SKB modules discovered at runtime: "
                + ", ".join(cleaned)
                + "."
            )
        else:
            module_schema.pop("enum", None)
            module_schema["description"] = "Optional SKB module name used to focus ranking"
            self.spec.description = self._BASE_SPEC.description

    async def refresh_modules(self, *, force_refresh: bool = False) -> list[str]:
        modules = await self.client.discover_modules(force_refresh=force_refresh)
        self.set_modules(modules)
        return self.modules

    async def execute(self, arguments: dict[str, Any]) -> AgentResult:
        query = str(arguments.get("query", "")).strip()
        module = str(arguments.get("module", "")).strip() or None
        try:
            limit = int(arguments.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5

        if not query:
            return AgentResult(ok=False, error="missing_query")

        try:
            results = await self.client.search(query, module=module, limit=limit)
        except Exception as exc:
            return AgentResult(ok=False, error=f"skb_search_error:{type(exc).__name__}")

        return AgentResult(
            ok=True,
            data={
                "query": query,
                "module": module,
                "available_modules": self.modules,
                "base_url": self.client.base_url,
                "count": len(results),
                "results": [
                    {
                        "title": item.title,
                        "url": item.url,
                        "snippet": item.snippet,
                        "score": item.score,
                    }
                    for item in results
                ],
            },
        )
