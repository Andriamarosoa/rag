from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from app.agents.email_agent import SendEmailAgent
from app.agents.registry import AgentRegistry
from app.codex.client import CodexMcpClient
from app.codex.strict_segmented_service import StrictSegmentedDecisionService
from app.config import Settings, settings
from app.context.manager import ContextManager
from app.grounded_answer import GroundedAnswerService
from app.knowledge.docx import DocxIngestionError, DocxIngestor
from app.ollama.client import OllamaNativeClient
from app.rules.engine import RuleEngine
from app.segmented_orchestrator import SegmentedOrchestrator
from app.sessions.store import SessionStore
from app.skb.dokuwiki import DokuWikiClient
from app.skb.embeddings import OllamaEmbeddingClient
from app.skb.indexer import SkbIndexer
from app.skb.models import SKB_MODULES, normalize_module_filter
from app.skb.retriever import SkbRetriever
from app.skb.vector_store import MariaDBVectorStore
from app.ws.manager import ConnectionManager
from app.ws.router import build_router


ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT_DIR / "index.html"
ACTION_TEMPLATE_JS = ROOT_DIR / "action-template.js"
KNOWLEDGE_UPLOAD_DIR = (
    settings.knowledge_upload_dir
    if settings.knowledge_upload_dir.is_absolute()
    else ROOT_DIR / settings.knowledge_upload_dir
).resolve()
docx_ingestor = DocxIngestor(
    max_bytes=settings.knowledge_upload_max_bytes,
    chunk_size=settings.skb_chunk_size,
    chunk_overlap=settings.skb_chunk_overlap,
)

store = SessionStore(
    host=settings.db_host,
    port=settings.db_port,
    user=settings.db_user,
    password=settings.db_password,
    database=settings.db_name,
    min_pool_size=settings.db_pool_min_size,
    max_pool_size=settings.db_pool_max_size,
    connect_timeout=settings.db_connect_timeout_seconds,
)
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
skb_wiki_client = DokuWikiClient(
    settings.skb_base_url,
    timeout_seconds=settings.skb_request_timeout_seconds,
    concurrency=settings.skb_crawl_concurrency,
)
skb_embedding_client = OllamaEmbeddingClient(
    settings.skb_embedding_base_url,
    model=settings.skb_embedding_model,
    timeout_seconds=settings.skb_embedding_timeout_seconds,
    batch_size=settings.skb_embedding_batch_size,
    dimension=settings.skb_embedding_dimension,
)
skb_vector_store = MariaDBVectorStore(
    host=settings.db_host,
    port=settings.db_port,
    user=settings.db_user,
    password=settings.db_password,
    database=settings.db_name,
    dimension=settings.skb_embedding_dimension,
    min_pool_size=settings.db_pool_min_size,
    max_pool_size=settings.db_pool_max_size,
    connect_timeout=settings.db_connect_timeout_seconds,
)
skb_indexer = SkbIndexer(
    skb_wiki_client,
    skb_embedding_client,
    skb_vector_store,
    chunk_size=settings.skb_chunk_size,
    chunk_overlap=settings.skb_chunk_overlap,
    min_chunk_size=settings.skb_min_chunk_size,
)
skb_retriever = SkbRetriever(
    skb_embedding_client,
    skb_vector_store,
    top_k=settings.skb_retrieval_top_k,
    max_distance=settings.skb_retrieval_max_distance,
    index_signature=skb_indexer.index_signature,
    source_base_url=settings.skb_base_url,
)
codex = StrictSegmentedDecisionService(codex_client, decision_client=ollama_decision_client)
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
grounded_answer = GroundedAnswerService(
    skb_retriever,
    ollama_decision_client,
    max_context_characters=settings.skb_answer_max_context_characters,
    ambiguity_distance_delta=settings.skb_ambiguity_distance_delta,
)
orchestrator = SegmentedOrchestrator(
    store,
    context_manager,
    rules,
    agents,
    codex,
    grounded_answer=grounded_answer,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_index_task: asyncio.Task[dict[str, Any]] | None = None
_index_schedule_lock = asyncio.Lock()
_index_state: dict[str, Any] = {
    "initialized": False,
    "running": False,
    "ready": False,
    "configured_signature": skb_indexer.index_signature,
    "active_generation_id": None,
    "active_signature": None,
    "last_started_at": None,
    "last_completed_at": None,
    "last_error": None,
    "last_result": None,
}


async def _refresh_index_readiness() -> None:
    active = await skb_vector_store.get_active_generation()
    active_signature = active.index_signature if active is not None else None
    _index_state.update(
        {
            "active_generation_id": (
                active.generation_id if active is not None else None
            ),
            "active_signature": active_signature,
            "ready": active_signature == skb_indexer.index_signature,
        }
    )


async def _ensure_vector_store() -> None:
    if _index_state["initialized"]:
        return
    await skb_vector_store.initialize()
    _index_state["initialized"] = True
    await _refresh_index_readiness()
    _index_state["last_error"] = None


async def _run_index_sync() -> dict[str, Any]:
    _index_state.update(
        {
            "running": True,
            "last_started_at": _utc_now(),
            "last_error": None,
        }
    )
    try:
        await _ensure_vector_store()
        result = await skb_indexer.sync()
        serialized = asdict(result)
        _index_state["last_result"] = serialized
        await _refresh_index_readiness()
        if result.failed_pages:
            _index_state["last_error"] = (
                f"partial_sync_failed: {result.failed_pages} page(s) failed"
            )
        elif not result.activated:
            _index_state["last_error"] = (
                f"index_activation_deferred: {result.activation_reason or 'unknown'}"
            )
        return serialized
    except asyncio.CancelledError:
        _index_state["last_error"] = "cancelled"
        raise
    except Exception as exc:
        _index_state["last_error"] = f"{type(exc).__name__}: {exc}"
        return {"error": _index_state["last_error"]}
    finally:
        _index_state["running"] = False
        _index_state["last_completed_at"] = _utc_now()


async def _start_index_sync() -> asyncio.Task[dict[str, Any]]:
    global _index_task
    async with _index_schedule_lock:
        if _index_task is None or _index_task.done():
            _index_task = asyncio.create_task(_run_index_sync(), name="skb-index-sync")
        return _index_task


async def _index_status() -> dict[str, Any]:
    if _index_state["initialized"]:
        try:
            await _refresh_index_readiness()
        except Exception as exc:
            _index_state["ready"] = False
            _index_state["last_error"] = f"{type(exc).__name__}: {exc}"
    payload = dict(_index_state)
    if _index_state["initialized"]:
        try:
            payload["store"] = asdict(await skb_vector_store.stats())
        except Exception as exc:
            payload["store_error"] = f"{type(exc).__name__}: {exc}"
    return payload


@asynccontextmanager
async def lifespan(_: FastAPI):
    await store.initialize()
    rules.reload()

    # External services must not keep the HTTP health endpoint from starting.  The
    # strict chat path will return a source-unavailable response rather than falling
    # back to model knowledge when the vector store or SKB cannot be reached.
    try:
        await _ensure_vector_store()
    except Exception as exc:
        _index_state["last_error"] = f"{type(exc).__name__}: {exc}"
    else:
        if settings.skb_sync_on_startup:
            await _start_index_sync()

    yield

    if _index_task is not None and not _index_task.done():
        _index_task.cancel()
        await asyncio.gather(_index_task, return_exceptions=True)
    await skb_wiki_client.close()
    await skb_embedding_client.close()
    await skb_vector_store.close()
    await store.close()
    await ollama_decision_client.close()
    await codex_client.close()


def _configure_cors(application: FastAPI, app_settings: Settings) -> None:
    allowed_origins = app_settings.cors_allow_origin_list
    allowed_origin_regex = app_settings.cors_allow_origin_regex.strip() or None
    if not allowed_origins and allowed_origin_regex is None:
        return

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=allowed_origin_regex,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


app = FastAPI(title="RAG WebSocket Orchestrator", version="0.1.0", lifespan=lifespan)
_configure_cors(app, settings)
app.include_router(build_router(manager, orchestrator, agents, rules))


@app.get("/", include_in_schema=False)
async def test_ui() -> HTMLResponse:
    html = INDEX_HTML.read_text(encoding="utf-8")
    asset_version = ACTION_TEMPLATE_JS.stat().st_mtime_ns
    script_tag = f'<script src="/action-template.js?v={asset_version}"></script>'
    if 'src="/action-template.js' not in html:
        html = html.replace("</body>", f"{script_tag}\n</body>")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/action-template.js", include_in_schema=False)
async def action_template_script() -> FileResponse:
    return FileResponse(
        ACTION_TEMPLATE_JS,
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "agents": [spec["name"] for spec in agents.specs()],
        "rules": len(rules.rules.rules),
        "assistant_decision_provider": "ollama_native",
        "assistant_decision_model": ollama_decision_client.model,
        "assistant_decision_thinking": False,
        "assistant_decision_pipeline": "skb_vector_retrieval_then_grounded_answer",
        "strict_skb_grounding": True,
        "rules_enabled_for_chat": False,
        "action_templates": True,
        "skb_base_url": skb_wiki_client.base_url,
        "skb_module_count": len(SKB_MODULES),
        "skb_embedding_model": skb_embedding_client.model,
        "skb_vector_store": "mariadb_vector_cosine",
        "skb_index_initialized": _index_state["initialized"],
        "skb_index_running": _index_state["running"],
        "skb_index_ready": _index_state["ready"],
        "skb_active_generation_id": _index_state["active_generation_id"],
        "skb_index_error": _index_state["last_error"],
    }


@app.get("/skb/modules")
async def skb_modules() -> dict:
    modules = [
        {
            "id": item.namespace,
            "label": item.label,
            "start_page_id": item.start_page_id,
            "url": skb_wiki_client.page_url(item.start_page_id),
        }
        for item in SKB_MODULES
    ]
    return {
        "base_url": skb_wiki_client.base_url,
        "count": len(modules),
        "modules": modules,
    }


def _public_file(record: dict[str, Any], *, created: bool | None = None) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"storage_key"}
    }
    payload["download_url"] = f"/knowledge/files/{record['file_id']}/download"
    if created is not None:
        payload["created"] = created
    return payload


@app.post("/knowledge/files", status_code=201)
async def upload_knowledge_file(
    module: str = Form(...),
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
) -> dict[str, Any]:
    namespace = module.strip().casefold()
    module_map = {item.namespace: item for item in SKB_MODULES}
    selected = module_map.get(namespace)
    if selected is None:
        raise HTTPException(status_code=400, detail={"code": "invalid_module"})

    try:
        data = await file.read(settings.knowledge_upload_max_bytes + 1)
    finally:
        await file.close()
    if len(data) > settings.knowledge_upload_max_bytes:
        raise HTTPException(status_code=413, detail={"code": "file_too_large"})
    try:
        document = docx_ingestor.ingest(
            data,
            filename=file.filename or "document.docx",
            content_type=file.content_type or "application/octet-stream",
            module_namespace=namespace,
            title=title,
        )
    except DocxIngestionError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_docx", "message": str(exc)},
        ) from exc

    try:
        await _ensure_vector_store()
        existing = await skb_vector_store.find_knowledge_file(
            namespace, document.file_sha256
        )
        if existing is not None:
            return _public_file(existing, created=False)
        vectors = await skb_embedding_client.embed_texts(
            [chunk.embedding_text for chunk in document.chunks]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "docx_embedding_failed", "message": str(exc)},
        ) from exc

    file_id = str(uuid4())
    storage_key = f"{file_id}.docx"
    target = (KNOWLEDGE_UPLOAD_DIR / storage_key).resolve()
    if target.parent != KNOWLEDGE_UPLOAD_DIR:
        raise HTTPException(status_code=500, detail={"code": "unsafe_storage_path"})
    temporary = target.with_suffix(".uploading")
    try:
        KNOWLEDGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(data)
        os.replace(temporary, target)
        stored_id, created = await skb_vector_store.add_knowledge_file(
            file_id=file_id,
            document=document,
            module=normalize_module_filter(namespace) or selected.label,
            storage_key=storage_key,
            index_signature=skb_indexer.index_signature,
            embeddings=vectors,
        )
        if not created:
            target.unlink(missing_ok=True)
        record = await skb_vector_store.get_knowledge_file(stored_id)
        if record is None:
            raise RuntimeError("stored DOCX metadata is unavailable")
        return _public_file(record, created=created)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if file_id == locals().get("stored_id", file_id):
            target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail={"code": "docx_storage_failed", "message": str(exc)},
        ) from exc


@app.get("/knowledge/files")
async def list_knowledge_files() -> dict[str, Any]:
    await _ensure_vector_store()
    records = await skb_vector_store.list_knowledge_files()
    return {"count": len(records), "files": [_public_file(row) for row in records]}


@app.get("/knowledge/files/{file_id}/download")
async def download_knowledge_file(file_id: UUID) -> FileResponse:
    await _ensure_vector_store()
    record = await skb_vector_store.get_knowledge_file(str(file_id))
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "file_not_found"})
    storage_key = str(record["storage_key"])
    target = (KNOWLEDGE_UPLOAD_DIR / storage_key).resolve()
    if target.parent != KNOWLEDGE_UPLOAD_DIR or not target.is_file():
        raise HTTPException(status_code=404, detail={"code": "file_content_not_found"})
    return FileResponse(
        target,
        media_type=str(record["mime_type"]),
        filename=str(record["original_filename"]),
    )


@app.delete("/knowledge/files/{file_id}")
async def delete_knowledge_file(file_id: UUID) -> dict[str, Any]:
    await _ensure_vector_store()
    deleted = await skb_vector_store.delete_knowledge_file(str(file_id))
    if deleted is None:
        raise HTTPException(status_code=404, detail={"code": "file_not_found"})
    target = (KNOWLEDGE_UPLOAD_DIR / str(deleted["storage_key"])).resolve()
    if target.parent == KNOWLEDGE_UPLOAD_DIR:
        target.unlink(missing_ok=True)
    return {"deleted": True, "file_id": str(file_id)}


@app.get("/skb/index/status")
async def skb_index_status() -> dict:
    return await _index_status()


@app.post("/skb/index/sync")
async def skb_index_sync(wait: bool = False) -> dict:
    task = await _start_index_sync()
    if wait:
        await task
    status = await _index_status()
    if wait and status.get("last_error"):
        raise HTTPException(
            status_code=502,
            detail={"code": "skb_index_sync_failed", "status": status},
        )
    return status


@app.get("/skb/search")
async def skb_search(
    q: str = Query(min_length=1),
    module: str | None = None,
    limit: int = Query(default=5, ge=1, le=10),
) -> dict:
    try:
        await _ensure_vector_store()
        request_retriever = SkbRetriever(
            skb_embedding_client,
            skb_vector_store,
            top_k=limit,
            max_distance=settings.skb_retrieval_max_distance,
            index_signature=skb_indexer.index_signature,
            source_base_url=settings.skb_base_url,
        )
        results = await request_retriever.retrieve(q, module=module)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "skb_vector_search_failed",
                "error": type(exc).__name__,
                "message": str(exc),
                "base_url": skb_wiki_client.base_url,
            },
        ) from exc
    return {
        "query": q,
        "module": module,
        "count": len(results),
        "results": [
            {
                "id": item.chunk_id,
                "page_id": item.page_id,
                "title": item.title,
                "url": item.source_url,
                "module": item.module,
                "section": item.section,
                "snippet": item.text,
                "score": item.score,
                "distance": item.distance,
                "source_kind": item.source_kind,
                "document_id": item.document_id,
            }
            for item in results
        ],
    }
