"""Artificer Routes -- Phase 10 of the IECP protocol.

Register, unregister, and list artificers.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...types.entity import EntityId
from ..errors import NotFoundError, ValidationError


@dataclass
class ArtificerRegistration:
    entity_id: EntityId
    persona: str | None
    model_config_data: dict[str, Any] | None
    registered_at: float


def _validate_required(body: dict, fields: list[str]) -> None:
    for name in fields:
        if body.get(name) is None:
            raise ValidationError(f"Missing required field: {name}")


def create_artificer_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def register_artificer(request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        _validate_required(body, ["entityId"])

        entity_id = EntityId(body["entityId"])
        registration = ArtificerRegistration(
            entity_id=entity_id,
            persona=body.get("persona"),
            model_config_data=body.get("modelConfig"),
            registered_at=time.time() * 1000,
        )

        services.artificer_registry[entity_id] = registration
        return JSONResponse(status_code=201, content=_registration_to_dict(registration))

    @router.delete("/{entity_id}")
    async def unregister_artificer(entity_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        eid = EntityId(entity_id)
        if eid not in services.artificer_registry:
            raise NotFoundError(f"Artificer not found: {entity_id}")

        del services.artificer_registry[eid]
        return JSONResponse(status_code=204, content=None)

    @router.get("")
    async def list_artificers(request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        registrations = list(services.artificer_registry.values())
        return JSONResponse(content=[_registration_to_dict(r) for r in registrations])

    return router


def _registration_to_dict(reg: ArtificerRegistration) -> dict[str, Any]:
    return {
        "entityId": reg.entity_id,
        "persona": reg.persona,
        "modelConfig": reg.model_config_data,
        "registeredAt": reg.registered_at,
    }
