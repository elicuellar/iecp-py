"""Attention Signal Routes -- Phase 10 of the IECP protocol.

Emit and query attention signals.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...types.entity import EntityId
from ...types.event import ConversationId
from ..errors import ValidationError


def _validate_required(body: dict, fields: list[str]) -> None:
    for name in fields:
        if body.get(name) is None:
            raise ValidationError(f"Missing required field: {name}")


def create_signal_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def emit_signal(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        _validate_required(body, ["entityId", "signalType"])

        accepted = services.signal_manager.signal(
            entity_id=EntityId(body["entityId"]),
            conversation_id=ConversationId(conv_id),
            signal_type=body["signalType"],
            note=body.get("note"),
        )

        if accepted:
            return JSONResponse(status_code=201, content={"accepted": True})
        else:
            return JSONResponse(status_code=429, content={"accepted": False, "reason": "Rate limited"})

    @router.get("")
    async def get_signals(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        signals = services.signal_manager.get_signals(ConversationId(conv_id))
        return JSONResponse(content=[_signal_to_dict(s) for s in signals])

    return router


def _signal_to_dict(signal: Any) -> dict[str, Any]:
    # Support both Pydantic models and dataclasses
    if hasattr(signal, "model_dump"):
        d = signal.model_dump()
    elif hasattr(signal, "__dataclass_fields__"):
        import dataclasses
        d = dataclasses.asdict(signal)
    else:
        d = dict(vars(signal))

    return {
        "entity_id": d.get("entity_id"),
        "conversation_id": d.get("conversation_id"),
        "signal_type": d.get("signal_type"),
        "note": d.get("note"),
        "batch_id": d.get("batch_id"),
        "created_at": d.get("created_at"),
        "expires_at": d.get("expires_at"),
    }
