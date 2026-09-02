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

                if event.type == "ping":
                    await manager.send(
                        session,
                        ServerEnvelope(type="pong", request_id=event.request_id, user_id=event.user_id).model_dump(),
                    )
                    continue

                if event.type == "rules.reload":
                    loaded = rules.reload()
                    await manager.send(
                        session,
                        ServerEnvelope(
                            type="rules.reloaded",
                            request_id=event.request_id,
                            user_id=event.user_id,
                            chat_id=event.chat_id,
                            data={"version": loaded.version, "count": len(loaded.rules)},
                        ).model_dump(),
                    )
                    continue

                if event.type == "agent.execute":
                    name = str(event.data.get("agent", ""))
                    arguments = event.data.get("arguments") or {}
                    confirmed = bool(event.data.get("confirmed", False))
                    result = await agents.execute(name, arguments, confirmed=confirmed)
                    await manager.send(
                        session,
                        ServerEnvelope(
                            type="agent.result",
                            request_id=event.request_id,
                            user_id=event.user_id,
                            chat_id=event.chat_id,
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
                            chat_id=event.chat_id,
                        ).model_dump(),
                    )
                    result = await orchestrator.handle_message(event.user_id, event.chat_id, text)
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
