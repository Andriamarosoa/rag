from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from fastapi import WebSocket


@dataclass(slots=True)
class WebSocketSession:
    connection_id: str
    websocket: WebSocket
    user_id: str | None = None
    chat_id: str | None = None
    tasks: set[asyncio.Task] = field(default_factory=set)


class ConnectionManager:
    def __init__(self):
        self.sessions: dict[str, WebSocketSession] = {}

    async def connect(self, websocket: WebSocket) -> WebSocketSession:
        await websocket.accept()
        connection_id = str(uuid4())
        session = WebSocketSession(connection_id=connection_id, websocket=websocket)
        self.sessions[connection_id] = session
        return session

    async def send(self, session: WebSocketSession, payload: dict) -> None:
        await session.websocket.send_json(payload)

    async def disconnect(self, session: WebSocketSession) -> None:
        for task in list(session.tasks):
            task.cancel()
        if session.tasks:
            await asyncio.gather(*session.tasks, return_exceptions=True)
        self.sessions.pop(session.connection_id, None)
