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
from app.ollama.client import OllamaNativeClient
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
ollama_decision_client = OllamaNativeClient(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model or settings.codex_model or "qwen3:8b",
)
codex = CodexService(codex_client, decision_client=ollama_decision_client)
context_manager = ContextManager(
    store=store,
    summarizer=codex,
    compact_at_tokens=settings.context_compact_at_tokens,
    keep_recent_tokens=settings.context_keep_recent_tokens,
    summary_target_tokens=settings.context_summary_target_tokens,
)
rules = RuleEngine(settings.rules_path)
agents = AgentRegistry()
agents.register(SendEmailAgent(settings))
manager = ConnectionManager()
orchestrator = Orchestrator(store, context_manager, rules, agents, codex)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await store.initialize()
    rules.reload()
    yield
    await ollama_decision_client.close()
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
        "assistant_decision_provider": "ollama_native",
        "assistant_decision_model": ollama_decision_client.model,
        "assistant_decision_thinking": False,
    }
