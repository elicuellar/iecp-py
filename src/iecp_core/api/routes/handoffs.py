"""Handoff Routes -- Phase 10 of the IECP protocol.

Create and list active handoffs.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...types.entity import EntityId
from ...types.event import ConversationId, EventId
from ...utils import generate_id
from ..errors import ValidationError


def _validate_required(body: dict, fields: list[str]) -> None:
    for name in fields:
        if body.get(name) is None:
            raise ValidationError(f"Missing required field: {name}")


def create_handoff_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def create_handoff(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        _validate_required(body, ["from_entity", "to_entity", "reason", "source_event"])

        result = services.handoff_manager.handoff(
            event_id=EventId(generate_id()),
            conversation_id=ConversationId(conv_id),
            from_entity=EntityId(body["from_entity"]),
            to_entity=EntityId(body["to_entity"]),
            reason=body["reason"],
            context_summary=body.get("context_summary", ""),
            source_event=EventId(body["source_event"]),
        )

        if result.get("success"):
            handoff = result["handoff"]
            return JSONResponse(status_code=201, content=_handoff_to_dict(handoff))
        else:
            return JSONResponse(
                status_code=409,
                content={"error": {"code": "HANDOFF_FAILED", "message": result.get("error")}},
            )

    @router.get("")
    async def list_handoffs(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        handoff = services.handoff_manager.get_active_handoff(ConversationId(conv_id))
        return JSONResponse(content=[_handoff_to_dict(handoff)] if handoff else [])

    return router


def _handoff_to_dict(handoff: Any) -> dict[str, Any]:
    if hasattr(handoff, "model_dump"):
        return handoff.model_dump()
    if hasattr(handoff, "__dataclass_fields__"):
        import dataclasses
        return dataclasses.asdict(handoff)
    return dict(vars(handoff))
