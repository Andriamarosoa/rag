from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from app.agents.email_agent import SendEmailAgent
from app.agents.registry import AgentRegistry
from app.agents.skb_search_agent import SkbSearchAgent
from app.codex.client import CodexMcpClient
from app.codex.segmented_service import SegmentedDecisionService
from app.config import settings
from app.context.manager import ContextManager
from app.ollama.client import OllamaNativeClient
from app.rules.engine import RuleEngine
from app.segmented_orchestrator import SegmentedOrchestrator
from app.sessions.store import SessionStore
from app.skb.client import SkbClient
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
skb_client = SkbClient(
    settings.skb_base_url,
    timeout_seconds=settings.skb_request_timeout_seconds,
    module_cache_seconds=settings.skb_module_cache_seconds,
    search_max_pages=settings.skb_search_max_pages,
)
skb_search_agent = SkbSearchAgent(skb_client)
codex = SegmentedDecisionService(codex_client, decision_client=ollama_decision_client)
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
agents.register(skb_search_agent)
manager = ConnectionManager()
orchestrator = SegmentedOrchestrator(store, context_manager, rules, agents, codex)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await store.initialize()
    rules.reload()

    # SKB is external runtime data. Failure to reach it must not prevent the assistant
    # from starting; when reachable, discovered modules become part of search_skb's
    # input schema, which is included in answer reasoning.
    try:
        await skb_search_agent.refresh_modules(force_refresh=True)
    except Exception:
        skb_search_agent.set_modules([])

    yield
    await skb_client.close()
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
        "assistant_decision_pipeline": "segmented_isolated_classifier_then_answer",
        "rule_classifier_context": "latest_user_segments_only",
        "segment_tracking": True,
        "skb_base_url": skb_client.base_url,
        "skb_module_count": len(skb_search_agent.modules),
    }


@app.get("/skb/modules")
async def skb_modules(refresh: bool = False) -> dict:
    try:
        modules = await skb_search_agent.refresh_modules(force_refresh=refresh)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "skb_unreachable",
                "error": type(exc).__name__,
                "message": str(exc),
                "base_url": skb_client.base_url,
            },
        ) from exc

    return {
        "base_url": skb_client.base_url,
        "count": len(modules),
        "modules": modules,
    }


@app.get("/skb/search")
async def skb_search(
    q: str = Query(min_length=1),
    module: str | None = None,
    limit: int = Query(default=5, ge=1, le=10),
) -> dict:
    result = await skb_search_agent.execute(
        {
            "query": q,
            "module": module,
            "limit": limit,
        }
    )
    if not result.ok:
        raise HTTPException(
            status_code=502,
            detail={
                "code": result.error or "skb_search_failed",
                "base_url": skb_client.base_url,
            },
        )
    return result.data
