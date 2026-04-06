"""Decision Routes -- Phase 10 of the IECP protocol.

Propose, update, and list decisions.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...types.entity import EntityId
from ...types.event import ConversationId, EventId
from ...utils import generate_id
from ..errors import NotFoundError, ValidationError


def _validate_required(body: dict, fields: list[str]) -> None:
    for name in fields:
        if body.get(name) is None:
            raise ValidationError(f"Missing required field: {name}")


def create_decision_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def propose_decision(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        _validate_required(body, ["summary", "proposed_by"])

        decision = services.decision_manager.propose(
            event_id=EventId(generate_id()),
            conversation_id=ConversationId(conv_id),
            summary=body["summary"],
            proposed_by=EntityId(body["proposed_by"]),
            context_events=body.get("context_events", []),
        )

        return JSONResponse(status_code=201, content=_decision_to_dict(decision))

    @router.patch("/{decision_id}")
    async def update_decision(conv_id: str, decision_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        action = body.get("action")

        if action == "affirm":
            _validate_required(body, ["entity_id"])
            decision = services.decision_manager.affirm(
                EventId(decision_id),
                EntityId(body["entity_id"]),
                body.get("is_human", True),
            )
        elif action == "reject":
            _validate_required(body, ["entity_id"])
            decision = services.decision_manager.reject(
                EventId(decision_id),
                EntityId(body["entity_id"]),
            )
        elif action == "supersede":
            _validate_required(body, ["summary", "proposed_by"])
            result = services.decision_manager.supersede(
                EventId(decision_id),
                event_id=EventId(generate_id()),
                conversation_id=ConversationId(conv_id),
                summary=body["summary"],
                proposed_by=EntityId(body["proposed_by"]),
                context_events=body.get("context_events", []),
            )
            if not result:
                raise NotFoundError(f"Decision not found: {decision_id}")
            return JSONResponse(content=_decision_to_dict(result["new"]))
        else:
            raise ValidationError("action must be one of: affirm, reject, supersede")

        if not decision:
            raise NotFoundError(f"Decision not found: {decision_id}")

        return JSONResponse(content=_decision_to_dict(decision))

    @router.get("")
    async def list_decisions(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        decisions = services.decision_manager.get_active_decisions(ConversationId(conv_id))
        return JSONResponse(content=[_decision_to_dict(d) for d in decisions])

    return router


def _decision_to_dict(decision: Any) -> dict[str, Any]:
    if hasattr(decision, "model_dump"):
        return decision.model_dump()
    if hasattr(decision, "__dataclass_fields__"):
        import dataclasses
        return dataclasses.asdict(decision)
    return dict(vars(decision))
