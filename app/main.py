from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.agents.email_agent import SendEmailAgent
from app.agents.registry import AgentRegistry
from app.codex.client import CodexMcpClient
from app.codex.service import CodexService
from app.config import settings
from app.context.manager import ContextManager
from app.orchestrator import Orchestrator
from app.rules.engine import RuleEngine
from app.sessions.store import SessionStore
from app.ws.manager import ConnectionManager
from app.ws.router import build_router


ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT_DIR / "index.html"

store = SessionStore(settings.database_path)
codex_client = CodexMcpClient(
    command=settings.codex_command,
    args=settings.codex_arg_list,
    cwd=settings.codex_cwd,
    start_tool=settings.codex_tool,
    reply_tool=settings.codex_reply_tool,
    model=settings.codex_model,
    sandbox=settings.codex_sandbox,
    approval_policy=settings.codex_approval_policy,
)
codex = CodexService(codex_client)
context_manager = ContextManager(
    store=store,
    summarizer=codex,
    compact_at_tokens=settings.context_compact_at_tokens,
    keep_recent_tokens=settings.context_keep_recent_tokens,
    summary_target_tokens=settings.context_summary_target_tokens,
)
rules = RuleEngine(settings.rules_path, codex)
agents = AgentRegistry()
agents.register(SendEmailAgent(settings))
manager = ConnectionManager()
orchestrator = Orchestrator(store, context_manager, rules, agents, codex)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await store.initialize()
    rules.reload()
    yield
    await codex_client.close()


app = FastAPI(title="RAG WebSocket Orchestrator", version="0.1.0", lifespan=lifespan)
app.include_router(build_router(manager, orchestrator, agents, rules))


@app.get("/", include_in_schema=False)
async def test_ui() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "agents": [spec["name"] for spec in agents.specs()],
        "rules": len(rules.rules.rules),
    }
