from __future__ import annotations

from typing import Any, Awaitable, Callable

FlowEmitter = Callable[[str, dict[str, Any]], Awaitable[None]]


async def emit_flow(
    emitter: FlowEmitter | None,
    event_type: str,
    **data: Any,
) -> None:
    if emitter is not None:
        await emitter(event_type, data)
