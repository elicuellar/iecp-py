"""Conversation Routes -- Phase 10 of the IECP protocol.

CRUD for conversations + participant management.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...types.entity import EntityId
from ...types.event import ConversationId
from ..errors import ConflictError, NotFoundError, ValidationError


def _validate_required(body: dict, fields: list[tuple[str, type]]) -> None:
    for name, expected_type in fields:
        val = body.get(name)
        if val is None:
            raise ValidationError(f"Missing required field: {name}")
        if not isinstance(val, expected_type):
            raise ValidationError(f"Field '{name}' must be of type {expected_type.__name__}")


def create_conversation_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def create_conversation(request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        _validate_required(body, [("created_by", str)])

        title = body.get("title", "")
        config_data = body.get("config")

        # Build config object if provided
        from ...types.conversation import ConversationConfig
        config: ConversationConfig | None = None
        if config_data and isinstance(config_data, dict):
            config = ConversationConfig(**config_data)

        conversation = await services.conversation_manager.create_conversation(
            title=title,
            created_by=EntityId(body["created_by"]),
            config=config,
        )

        return JSONResponse(status_code=201, content=_conv_to_dict(conversation))

    @router.get("/{conv_id}")
    async def get_conversation(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        conversation = await services.conversation_manager.get_conversation(ConversationId(conv_id))
        if not conversation:
            raise NotFoundError(f"Conversation not found: {conv_id}")

        participants = await services.conversation_manager.get_participants(ConversationId(conv_id))
        result = _conv_to_dict(conversation)
        result["participants"] = [_participant_to_dict(p) for p in participants]
        return JSONResponse(content=result)

    @router.patch("/{conv_id}")
    async def update_conversation(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()

        conversation = await services.conversation_manager.get_conversation(ConversationId(conv_id))
        if not conversation:
            raise NotFoundError(f"Conversation not found: {conv_id}")

        if "config" in body and isinstance(body["config"], dict):
            conversation = await services.conversation_manager.update_config(
                ConversationId(conv_id),
                body["config"],
            )

        return JSONResponse(content=_conv_to_dict(conversation))

    @router.post("/{conv_id}/participants")
    async def add_participant(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        _validate_required(body, [("entity_id", str)])

        role = body.get("role", "member")

        try:
            conversation = await services.conversation_manager.add_participant(
                ConversationId(conv_id),
                EntityId(body["entity_id"]),
                role,
            )
        except ValueError as e:
            if "already a participant" in str(e):
                raise ConflictError(str(e))
            raise

        # Return the newly added participant
        participants = await services.conversation_manager.get_participants(ConversationId(conv_id))
        new_participant = next(
            (p for p in participants if p.entity_id == body["entity_id"]), None
        )
        if new_participant:
            return JSONResponse(status_code=201, content=_participant_to_dict(new_participant))

        return JSONResponse(status_code=201, content={"entity_id": body["entity_id"], "role": role})

    @router.delete("/{conv_id}/participants/{entity_id}")
    async def remove_participant(conv_id: str, entity_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        await services.conversation_manager.remove_participant(
            ConversationId(conv_id),
            EntityId(entity_id),
        )
        return JSONResponse(status_code=204, content=None)

    @router.get("/{conv_id}/participants")
    async def get_participants(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        participants = await services.conversation_manager.get_participants(ConversationId(conv_id))
        return JSONResponse(content=[_participant_to_dict(p) for p in participants])

    return router


def _conv_to_dict(conv: Any) -> dict[str, Any]:
    d = conv.model_dump()
    return {
        "id": d["id"],
        "title": d.get("title", ""),
        "status": d.get("status", "active"),
        "config": d.get("config", {}),
        "created_by": d.get("created_by"),
        "created_at": d.get("created_at"),
        "updated_at": d.get("updated_at"),
    }


def _participant_to_dict(p: Any) -> dict[str, Any]:
    d = p.model_dump()
    return {
        "entity_id": d.get("entity_id"),
        "conversation_id": d.get("conversation_id"),
        "role": d.get("role"),
        "lifecycle_status": d.get("lifecycle_status"),
        "joined_at": d.get("joined_at"),
    }
