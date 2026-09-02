from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.agents.registry import AgentRegistry
from app.orchestrator import Orchestrator
from app.rules.engine import RuleEngine

from .manager import ConnectionManager
from .protocol import ClientEnvelope, ServerEnvelope


def build_router(
    manager: ConnectionManager,
    orchestrator: Orchestrator,
    agents: AgentRegistry,
    rules: RuleEngine,
) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        session = await manager.connect(websocket)
        try:
            await manager.send(
                session,
                ServerEnvelope(
                    type="connection.ready",
                    data={"connection_id": session.connection_id},
                ).model_dump(),
            )

            while True:
                raw = await websocket.receive_json()
                try:
                    event = ClientEnvelope.model_validate(raw)
                except ValidationError as exc:
                    await manager.send(
                        session,
                        ServerEnvelope(type="error", data={"code": "invalid_message", "details": exc.errors()}).model_dump(),
                    )
                    continue

                session.user_id = event.user_id
                if event.chat_id:
                    session.chat_id = event.chat_id

                async def send_flow_event(event_type: str, data: dict) -> None:
                    event_chat_id = data.get("chat_id")
                    if event_chat_id:
                        session.chat_id = str(event_chat_id)
                    await manager.send(
                        session,
                        ServerEnvelope(
                            type=event_type,
                            request_id=event.request_id,
                            user_id=event.user_id,
                            chat_id=session.chat_id,
                            data=data,
                        ).model_dump(),
                    )

                if event.type == "ping":
                    await manager.send(
                        session,
                        ServerEnvelope(type="pong", request_id=event.request_id, user_id=event.user_id).model_dump(),
                    )
                    continue

                if event.type == "rules.reload":
                    await send_flow_event("rules.reload.started", {})
                    try:
                        loaded = rules.reload()
                    except Exception as exc:
                        await send_flow_event(
                            "rules.reload.failed",
                            {"error": type(exc).__name__, "message": str(exc)},
                        )
                        continue
                    await manager.send(
                        session,
                        ServerEnvelope(
                            type="rules.reloaded",
                            request_id=event.request_id,
                            user_id=event.user_id,
                            chat_id=session.chat_id,
                            data={"version": loaded.version, "count": len(loaded.rules)},
                        ).model_dump(),
                    )
                    continue

                if event.type == "agent.execute":
                    name = str(event.data.get("agent", ""))
                    arguments = event.data.get("arguments") or {}
                    confirmed = bool(event.data.get("confirmed", False))
                    agent = agents.get(name)
                    await send_flow_event(
                        "agent.execution.started",
                        {
                            "agent": name,
                            "arguments": arguments,
                            "confirmed": confirmed,
                            "requires_confirmation": agent.spec.requires_confirmation if agent else None,
                        },
                    )
                    try:
                        result = await agents.execute(name, arguments, confirmed=confirmed)
                    except Exception as exc:
                        await send_flow_event(
                            "agent.execution.failed",
                            {"agent": name, "error": type(exc).__name__, "message": str(exc)},
                        )
                        continue
                    await send_flow_event(
                        "agent.execution.completed",
                        {
                            "agent": name,
                            "ok": result.ok,
                            "data": result.data,
                            "error": result.error,
                        },
                    )
                    await manager.send(
                        session,
                        ServerEnvelope(
                            type="agent.result",
                            request_id=event.request_id,
                            user_id=event.user_id,
                            chat_id=session.chat_id,
                            data={"agent": name, "ok": result.ok, "data": result.data, "error": result.error},
                        ).model_dump(),
                    )
                    continue

                if event.type == "chat.message":
                    text = str(event.data.get("text", "")).strip()
                    if not text:
                        await manager.send(
                            session,
                            ServerEnvelope(type="error", request_id=event.request_id, data={"code": "empty_message"}).model_dump(),
                        )
                        continue

                    await manager.send(
                        session,
                        ServerEnvelope(
                            type="assistant.started",
                            request_id=event.request_id,
                            user_id=event.user_id,
                            chat_id=session.chat_id,
                        ).model_dump(),
                    )
                    try:
                        result = await orchestrator.handle_message(
                            event.user_id,
                            event.chat_id,
                            text,
                            emit=send_flow_event,
                        )
                    except Exception as exc:
                        await send_flow_event(
                            "flow.failed",
                            {"error": type(exc).__name__, "message": str(exc)},
                        )
                        await manager.send(
                            session,
                            ServerEnvelope(
                                type="error",
                                request_id=event.request_id,
                                user_id=event.user_id,
                                chat_id=session.chat_id,
                                data={"code": "flow_failed", "message": str(exc)},
                            ).model_dump(),
                        )
                        continue

                    session.chat_id = result["chat_id"]
                    await manager.send(
                        session,
                        ServerEnvelope(
                            type="assistant.completed",
                            request_id=event.request_id,
                            user_id=event.user_id,
                            chat_id=result["chat_id"],
                            data=result,
                        ).model_dump(),
                    )

        except WebSocketDisconnect:
            pass
        finally:
            await manager.disconnect(session)

    return router
