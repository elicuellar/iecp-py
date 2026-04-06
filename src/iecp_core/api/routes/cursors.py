"""Cursor Routes -- Phase 10 of the IECP protocol.

Get and advance per-entity cursors in conversations.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...types.entity import EntityId
from ...types.event import ConversationId, EventId


def create_cursor_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/{entity_id}")
    async def get_cursor(conv_id: str, entity_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        cursor = await services.cursor_manager.get_cursor(
            EntityId(entity_id), ConversationId(conv_id)
        )
        return JSONResponse(content=_cursor_to_dict(cursor))

    @router.put("/{entity_id}")
    async def update_cursor(conv_id: str, entity_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        cursor = await services.cursor_manager.get_cursor(
            EntityId(entity_id), ConversationId(conv_id)
        )

        if body.get("received"):
            cursor = await services.cursor_manager.advance_received(
                EntityId(entity_id),
                ConversationId(conv_id),
                EventId(body["received"]),
            )

        if body.get("processed"):
            cursor = await services.cursor_manager.advance_processed(
                EntityId(entity_id),
                ConversationId(conv_id),
                EventId(body["processed"]),
            )

        return JSONResponse(content=_cursor_to_dict(cursor))

    return router


def _cursor_to_dict(cursor: Any) -> dict[str, Any]:
    if cursor is None:
        return {}
    if hasattr(cursor, "model_dump"):
        return cursor.model_dump()
    return dict(cursor) if cursor else {}
