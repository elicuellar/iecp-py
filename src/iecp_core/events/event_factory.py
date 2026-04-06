from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from ..types import (
    ActionContent,
    AttentionContent,
    AuthorType,
    BatchId,
    ConversationId,
    DecisionContent,
    EntityId,
    Event,
    EventId,
    HandoffContent,
    MessageContent,
    SystemContent,
)
from ..utils import generate_id


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_message_event(
    conversation_id: ConversationId,
    author_id: EntityId,
    author_type: AuthorType,
    text: str,
    format: Literal["plain", "markdown"] = "plain",
    mentions: list[EntityId] | None = None,
    parent_id: EventId | None = None,
    batch_id: BatchId | None = None,
    is_continuation: bool = False,
    is_complete: bool = True,
    ai_depth_counter: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Event:
    return Event(
        id=EventId(generate_id()),
        conversation_id=conversation_id,
        type="message",
        author_id=author_id,
        author_type=author_type,
        content=MessageContent(
            text=text,
            format=format,
            mentions=mentions or [],
        ),
        parent_id=parent_id,
        batch_id=batch_id,
        is_continuation=is_continuation,
        is_complete=is_complete,
        ai_depth_counter=ai_depth_counter,
        status="active",
        created_at=_now_iso(),
        metadata=metadata or {},
    )


def create_action_event(
    conversation_id: ConversationId,
    author_id: EntityId,
    author_type: AuthorType,
    action_type: str,
    description: str,
    result: str | None = None,
    action_status: Literal["pending", "completed", "failed"] = "pending",
    parent_id: EventId | None = None,
    batch_id: BatchId | None = None,
    is_continuation: bool = False,
    is_complete: bool = True,
    ai_depth_counter: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Event:
    return Event(
        id=EventId(generate_id()),
        conversation_id=conversation_id,
        type="action",
        author_id=author_id,
        author_type=author_type,
        content=ActionContent(
            action_type=action_type,
            description=description,
            result=result,
            status=action_status,
        ),
        parent_id=parent_id,
        batch_id=batch_id,
        is_continuation=is_continuation,
        is_complete=is_complete,
        ai_depth_counter=ai_depth_counter,
        status="active",
        created_at=_now_iso(),
        metadata=metadata or {},
    )


def create_system_event(
    conversation_id: ConversationId,
    system_event: str,
    description: str,
    data: dict[str, Any] | None = None,
    parent_id: EventId | None = None,
    batch_id: BatchId | None = None,
    metadata: dict[str, Any] | None = None,
) -> Event:
    return Event(
        id=EventId(generate_id()),
        conversation_id=conversation_id,
        type="system",
        author_id=EntityId("system"),
        author_type="system",
        content=SystemContent(
            system_event=system_event,
            description=description,
            data=data or {},
        ),
        parent_id=parent_id,
        batch_id=batch_id,
        is_continuation=False,
        is_complete=True,
        ai_depth_counter=0,
        status="active",
        created_at=_now_iso(),
        metadata=metadata or {},
    )


def create_attention_event(
    conversation_id: ConversationId,
    author_id: EntityId,
    author_type: AuthorType,
    signal: Literal["ping", "urgent", "fyi"],
    utterance_ref: EventId | None = None,
    note: str | None = None,
    parent_id: EventId | None = None,
    batch_id: BatchId | None = None,
    metadata: dict[str, Any] | None = None,
) -> Event:
    return Event(
        id=EventId(generate_id()),
        conversation_id=conversation_id,
        type="attention",
        author_id=author_id,
        author_type=author_type,
        content=AttentionContent(
            signal=signal,
            utterance_ref=utterance_ref,
            note=note,
        ),
        parent_id=parent_id,
        batch_id=batch_id,
        is_continuation=False,
        is_complete=True,
        ai_depth_counter=0,
        status="active",
        created_at=_now_iso(),
        metadata=metadata or {},
    )


def create_decision_event(
    conversation_id: ConversationId,
    author_id: EntityId,
    author_type: AuthorType,
    summary: str,
    proposed_by: EntityId,
    affirmed_by: list[EntityId] | None = None,
    context_events: list[EventId] | None = None,
    decision_status: Literal["proposed", "affirmed", "rejected"] = "proposed",
    parent_id: EventId | None = None,
    batch_id: BatchId | None = None,
    metadata: dict[str, Any] | None = None,
) -> Event:
    return Event(
        id=EventId(generate_id()),
        conversation_id=conversation_id,
        type="decision",
        author_id=author_id,
        author_type=author_type,
        content=DecisionContent(
            summary=summary,
            proposed_by=proposed_by,
            affirmed_by=affirmed_by or [],
            context_events=context_events or [],
            status=decision_status,
        ),
        parent_id=parent_id,
        batch_id=batch_id,
        is_continuation=False,
        is_complete=True,
        ai_depth_counter=0,
        status="active",
        created_at=_now_iso(),
        metadata=metadata or {},
    )


def create_handoff_event(
    conversation_id: ConversationId,
    author_id: EntityId,
    author_type: AuthorType,
    from_entity: EntityId,
    to_entity: EntityId,
    reason: str,
    context_summary: str,
    source_event: EventId | None = None,
    parent_id: EventId | None = None,
    batch_id: BatchId | None = None,
    metadata: dict[str, Any] | None = None,
) -> Event:
    return Event(
        id=EventId(generate_id()),
        conversation_id=conversation_id,
        type="handoff",
        author_id=author_id,
        author_type=author_type,
        content=HandoffContent(
            from_entity=from_entity,
            to_entity=to_entity,
            reason=reason,
            context_summary=context_summary,
            source_event=source_event,
        ),
        parent_id=parent_id,
        batch_id=batch_id,
        is_continuation=False,
        is_complete=True,
        ai_depth_counter=0,
        status="active",
        created_at=_now_iso(),
        metadata=metadata or {},
    )
