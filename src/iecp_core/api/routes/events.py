"""Event Routes -- Phase 10 of the IECP protocol.

Append, read, edit, and soft-delete events in the event log.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...events.event_factory import (
    create_action_event,
    create_attention_event,
    create_decision_event,
    create_handoff_event,
    create_message_event,
    create_system_event,
)
from ...events.event_store import ReadEventsOptions
from ...types.entity import EntityId
from ...types.event import ConversationId, EventId
from ..errors import NotFoundError, ValidationError


def _validate_required(body: dict, fields: list[tuple[str, type]]) -> None:
    for name, expected_type in fields:
        val = body.get(name)
        if val is None:
            raise ValidationError(f"Missing required field: {name}")
        if not isinstance(val, expected_type):
            raise ValidationError(f"Field '{name}' must be of type {expected_type.__name__}")


def create_event_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def append_event(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        body = await request.json()
        _validate_required(body, [
            ("type", str),
            ("author_id", str),
            ("author_type", str),
            ("content", dict),
        ])

        event_type = body["type"]
        author_id = EntityId(body["author_id"])
        author_type = body["author_type"]
        content = body["content"]
        conversation_id = ConversationId(conv_id)

        base_kwargs: dict[str, Any] = {
            "conversation_id": conversation_id,
            "author_id": author_id,
            "author_type": author_type,
            "is_continuation": body.get("is_continuation", False),
            "is_complete": body.get("is_complete", True),
            "ai_depth_counter": body.get("ai_depth_counter", 0),
            "metadata": body.get("metadata", {}),
        }

        if event_type == "message":
            event = create_message_event(
                **base_kwargs,
                text=content.get("text", ""),
                format=content.get("format", "plain"),
                mentions=content.get("mentions", []),
            )
        elif event_type == "action":
            event = create_action_event(
                **base_kwargs,
                action_type=content.get("action_type", ""),
                description=content.get("description", ""),
                result=content.get("result"),
                action_status=content.get("status", "pending"),
            )
        elif event_type == "system":
            event = create_system_event(
                conversation_id=conversation_id,
                system_event=content.get("system_event", ""),
                description=content.get("description", ""),
                data=content.get("data"),
                metadata=body.get("metadata", {}),
            )
        elif event_type == "attention":
            event = create_attention_event(
                **base_kwargs,
                signal=content.get("signal", "ping"),
                utterance_ref=EventId(content["utterance_ref"]) if content.get("utterance_ref") else None,
                note=content.get("note"),
            )
        elif event_type == "decision":
            event = create_decision_event(
                **base_kwargs,
                summary=content.get("summary", ""),
                proposed_by=EntityId(content.get("proposed_by", author_id)),
                affirmed_by=content.get("affirmed_by"),
                context_events=content.get("context_events"),
                decision_status=content.get("status", "proposed"),
            )
        elif event_type == "handoff":
            event = create_handoff_event(
                **base_kwargs,
                from_entity=EntityId(content.get("from_entity", author_id)),
                to_entity=EntityId(content["to_entity"]) if content.get("to_entity") else author_id,
                reason=content.get("reason", ""),
                context_summary=content.get("context_summary", ""),
                source_event=EventId(content["source_event"]) if content.get("source_event") else None,
            )
        else:
            raise ValidationError(f"Unknown event type: {event_type}")

        persisted = await services.event_store.append(event)

        # Broadcast via gateway if available
        if services.gateway:
            services.gateway.handle_event(persisted)

        # Feed to orchestrator pipeline
        if services.orchestrator:
            await services.orchestrator.handle_incoming_event(persisted)

        return JSONResponse(status_code=201, content=_event_to_dict(persisted))

    @router.get("")
    async def read_events(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        after = request.query_params.get("after")
        limit_str = request.query_params.get("limit", "50")
        limit = int(limit_str) if limit_str else 50

        options = ReadEventsOptions(
            after=EventId(after) if after else None,
            limit=limit,
        )

        result = await services.event_store.read_events(ConversationId(conv_id), options)
        return JSONResponse(content={
            "events": [_event_to_dict(e) for e in result.events],
            "has_more": result.has_more,
        })

    @router.get("/{event_id}")
    async def get_event(conv_id: str, event_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        event = await services.event_store.get_by_id(EventId(event_id))
        if not event:
            raise NotFoundError(f"Event not found: {event_id}")

        return JSONResponse(content=_event_to_dict(event))

    @router.patch("/{event_id}")
    async def edit_event(conv_id: str, event_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        event = await services.event_store.get_by_id(EventId(event_id))
        if not event:
            raise NotFoundError(f"Event not found: {event_id}")

        await services.event_store.update_status(EventId(event_id), "edited")
        updated = await services.event_store.get_by_id(EventId(event_id))
        return JSONResponse(content=_event_to_dict(updated))

    @router.delete("/{event_id}")
    async def delete_event(conv_id: str, event_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        event = await services.event_store.get_by_id(EventId(event_id))
        if not event:
            raise NotFoundError(f"Event not found: {event_id}")

        await services.event_store.update_status(EventId(event_id), "deleted")
        return JSONResponse(status_code=204, content=None)

    return router


def _event_to_dict(event: Any) -> dict[str, Any]:
    """Convert Event to API response dict (TypeScript-compatible)."""
    d = event.model_dump()
    return {
        "event_id": d["id"],
        "event_type": d["type"],
        "conversation_id": d["conversation_id"],
        "author_id": d["author_id"],
        "author_type": d["author_type"],
        "content": d["content"],
        "status": d.get("status", "active"),
        "is_continuation": d.get("is_continuation", False),
        "is_complete": d.get("is_complete", True),
        "ai_depth_counter": d.get("ai_depth_counter", 0),
        "batch_id": d.get("batch_id"),
        "parent_id": d.get("parent_id"),
        "created_at": d.get("created_at"),
        "metadata": d.get("metadata", {}),
    }
