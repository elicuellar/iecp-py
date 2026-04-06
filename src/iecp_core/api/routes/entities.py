"""Entity Routes -- Phase 10 of the IECP protocol.

CRUD for entities (humans, artificers, daemons).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...types.entity import EntityId
from ..errors import NotFoundError, ValidationError


VALID_ENTITY_TYPES = {"human", "artificer", "daemon"}


def _validate_required(body: dict, fields: list[tuple[str, type]]) -> None:
    for name, expected_type in fields:
        val = body.get(name)
        if val is None:
            raise ValidationError(f"Missing required field: {name}")
        if not isinstance(val, expected_type):
            raise ValidationError(f"Field '{name}' must be of type {expected_type.__name__}")


def create_entity_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def create_entity(request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        _validate_required(body, [("entity_type", str), ("display_name", str)])

        entity_type = body["entity_type"]
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValidationError("entity_type must be one of: human, artificer, daemon")

        entity = await services.entity_manager.create_entity(
            name=body["display_name"],
            type=entity_type,
        )

        return JSONResponse(
            status_code=201,
            content=_entity_to_dict(entity),
        )

    @router.get("")
    async def list_entities(request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        type_filter = request.query_params.get("type")
        entities = await services.entity_repo.list()

        if type_filter and type_filter in VALID_ENTITY_TYPES:
            entities = [e for e in entities if e.type == type_filter]

        return JSONResponse(content=[_entity_to_dict(e) for e in entities])

    @router.get("/{entity_id}")
    async def get_entity(entity_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        entity = await services.entity_manager.get_entity(EntityId(entity_id))
        if not entity:
            raise NotFoundError(f"Entity not found: {entity_id}")

        return JSONResponse(content=_entity_to_dict(entity))

    @router.patch("/{entity_id}")
    async def update_entity(entity_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()

        entity = await services.entity_manager.get_entity(EntityId(entity_id))
        if not entity:
            raise NotFoundError(f"Entity not found: {entity_id}")

        updates: dict[str, Any] = {}
        if "display_name" in body:
            if not isinstance(body["display_name"], str):
                raise ValidationError("Field 'display_name' must be of type str")
            updates["name"] = body["display_name"]
        if "model_info" in body:
            updates["model_info"] = body["model_info"]
        if "avatar_url" in body:
            updates["avatar_url"] = body["avatar_url"]

        updated = await services.entity_manager.update_entity(EntityId(entity_id), updates)
        return JSONResponse(content=_entity_to_dict(updated))

    return router


def _entity_to_dict(entity: Any) -> dict[str, Any]:
    """Convert an Entity to the API response dict (TypeScript-compatible field names)."""
    d = entity.model_dump()
    # Remap Python field names to TS-compatible API names
    result: dict[str, Any] = {
        "entity_id": d["id"],
        "entity_type": d["type"],
        "display_name": d["name"],
        "capabilities": d.get("capabilities", {}),
        "created_at": d.get("created_at"),
        "updated_at": d.get("updated_at"),
        "metadata": d.get("metadata", {}),
    }
    return result
