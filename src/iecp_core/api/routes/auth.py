"""Auth Routes -- Phase 10 of the IECP protocol.

Token generation for WebSocket authentication.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...gateway.types import AuthToken
from ...types.entity import EntityId
from ...types.event import ConversationId
from ...utils import generate_id
from ..errors import ValidationError


def _validate_required(body: dict, fields: list[str]) -> None:
    for name in fields:
        if body.get(name) is None:
            raise ValidationError(f"Missing required field: {name}")


def create_auth_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/tokens")
    async def generate_token(request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        _validate_required(body, ["entityId", "type"])

        entity_id = EntityId(body["entityId"])
        token_type = body["type"]
        conversation_ids = [ConversationId(c) for c in body.get("conversationIds", [])]

        token = f"iecp_ws_{generate_id()}"

        services.token_validator.add_token(
            token,
            AuthToken(
                entity_id=entity_id,
                type=token_type,
                conversation_ids=conversation_ids,
            ),
        )

        return JSONResponse(
            status_code=201,
            content={
                "token": token,
                "entityId": entity_id,
                "type": token_type,
                "conversationIds": list(conversation_ids),
            },
        )

    return router
