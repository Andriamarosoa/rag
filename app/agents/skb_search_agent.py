from __future__ import annotations

from typing import Any

from app.skb.client import SkbClient

from .base import AgentResult, AgentSpec, CodeAgent


class SkbSearchAgent(CodeAgent):
    spec = AgentSpec(
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
