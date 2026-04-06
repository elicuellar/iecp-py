"""Floor Lock Routes -- Phase 10 of the IECP protocol.

Acquire, release, and query lock state.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...lock.types import LockRequest
from ...types.entity import EntityId
from ...types.event import ConversationId
from ..errors import ValidationError


def _validate_required(body: dict, fields: list[str]) -> None:
    for name in fields:
        if body.get(name) is None:
            raise ValidationError(f"Missing required field: {name}")


def create_lock_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/acquire")
    async def acquire_lock(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        _validate_required(body, ["entityId"])

        result = services.floor_lock.acquire(
            LockRequest(
                entity_id=EntityId(body["entityId"]),
                conversation_id=ConversationId(conv_id),
                estimated_ms=body.get("estimatedMs", 30000),
                priority="default",
            )
        )

        if result.granted:
            lock_dict = result.lock.model_dump() if result.lock else None
            return JSONResponse(content={"granted": True, "lock": lock_dict})
        else:
            return JSONResponse(content={
                "granted": False,
                "reason": result.reason,
                "queue_position": result.queue_position,
            })

    @router.post("/release")
    async def release_lock(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        _validate_required(body, ["entityId"])

        services.floor_lock.release(
            ConversationId(conv_id),
            EntityId(body["entityId"]),
            "commit",
        )
        return JSONResponse(content={"released": True})

    @router.get("")
    async def get_lock_state(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        is_locked = services.floor_lock.is_locked(ConversationId(conv_id))
        state = services.floor_lock.get_lock_state(ConversationId(conv_id))

        state_dict = state.model_dump() if state else None
        return JSONResponse(content={"locked": is_locked, "state": state_dict})

    return router
