from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass(slots=True)
class CodexResult:
    text: str
    thread_id: str | None = None
    raw: Any = None


class CodexMcpClient:
    """Small adapter around the legacy Codex MCP server.

    OpenAI deprecated `codex mcp-server` on 2026-08-24 in favor of Codex App
    Server. Keeping the adapter isolated lets the application migrate transports
    without changing WebSocket, session, context, rules, or code-agent layers.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        cwd: Path,
        start_tool: str = "codex",
        reply_tool: str = "codex-reply",
        model: str = "",
        sandbox: str = "read-only",
        approval_policy: str = "never",
    ):
        self.command = command
        self.args = args
        self.cwd = cwd
        self.start_tool = start_tool
        self.reply_tool = reply_tool
        self.model = model
        self.sandbox = sandbox
        self.approval_policy = approval_policy
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._session:
            return
        async with self._connect_lock:
            if self._session:
                return
            stack = AsyncExitStack()
            params = StdioServerParameters(
                command=self.command,
                args=self.args,
                cwd=str(self.cwd),
            )
            read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self._stack = stack
            self._session = session

    async def close(self) -> None:
        if self._stack:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def list_tools(self) -> list[str]:
        await self.connect()
        assert self._session
        result = await self._session.list_tools()
        return [tool.name for tool in result.tools]

    async def ask(self, prompt: str, thread_id: str | None = None) -> CodexResult:
        await self.connect()
        assert self._session

        if thread_id:
            tool = self.reply_tool
            # Codex reply schema is camelCase.
            arguments: dict[str, Any] = {"threadId": thread_id, "prompt": prompt}
        else:
            tool = self.start_tool
            arguments = {
                "prompt": prompt,
                "cwd": str(self.cwd.resolve()),
                "sandbox": self.sandbox,
                "approval-policy": self.approval_policy,
            }
            if self.model:
                arguments["model"] = self.model

        result = await self._session.call_tool(tool, arguments=arguments)
        structured = getattr(result, "structured_content", None)
        if structured is None:
            structured = getattr(result, "structuredContent", None)

        text = ""
        parsed_thread = thread_id
        if isinstance(structured, dict):
            text = str(structured.get("content") or "")
            parsed_thread = str(structured.get("threadId") or structured.get("thread_id") or parsed_thread or "") or None

        if not text:
            text_parts: list[str] = []
            for item in getattr(result, "content", []) or []:
                item_text = getattr(item, "text", None)
                if item_text:
                    text_parts.append(item_text)
            text = "\n".join(text_parts).strip()

        if not parsed_thread:
            parsed_thread = self._extract_thread_id(text)

        return CodexResult(text=text.strip(), thread_id=parsed_thread, raw=result)

    @staticmethod
    def _extract_thread_id(text: str) -> str | None:
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                value = payload.get("threadId") or payload.get("thread_id")
                return str(value) if value else None
        except Exception:
            pass
        return None
