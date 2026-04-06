from __future__ import annotations

from typing import Any, Literal, NewType, Union

from pydantic import BaseModel

from .entity import EntityId

EventId = NewType("EventId", str)
ConversationId = NewType("ConversationId", str)
BatchId = NewType("BatchId", str)
EventType = Literal[
    "message",
    "action",
    "system",
    "attention",
    "decision",
    "handoff",
    "lock_acquired",
    "lock_released",
]
EventStatus = Literal["active", "edited", "deleted", "interrupted"]
AuthorType = Literal["human", "artificer", "daemon", "system"]


class MessageContent(BaseModel):
    text: str
    format: Literal["plain", "markdown"] = "plain"
    mentions: list[EntityId] = []


class ActionContent(BaseModel):
    action_type: str
    description: str
    result: str | None = None
    status: Literal["pending", "completed", "failed"] = "pending"


class SystemContent(BaseModel):
    system_event: str
    description: str
    data: dict[str, Any] = {}


class AttentionContent(BaseModel):
    signal: Literal["ping", "urgent", "fyi"]
    utterance_ref: EventId | None = None
    note: str | None = None


class DecisionContent(BaseModel):
    summary: str
    proposed_by: EntityId
    affirmed_by: list[EntityId] = []
    context_events: list[EventId] = []
    status: Literal["proposed", "affirmed", "rejected"] = "proposed"


class HandoffContent(BaseModel):
    from_entity: EntityId
    to_entity: EntityId
    reason: str
    context_summary: str
    source_event: EventId | None = None


EventContent = Union[
    MessageContent,
    ActionContent,
    SystemContent,
    AttentionContent,
    DecisionContent,
    HandoffContent,
]


class Event(BaseModel):
    id: EventId
    conversation_id: ConversationId
    type: EventType
    author_id: EntityId
    author_type: AuthorType
    content: EventContent
    parent_id: EventId | None = None
    batch_id: BatchId | None = None
    is_continuation: bool = False
    is_complete: bool = True
    ai_depth_counter: int = 0
    status: EventStatus = "active"
    created_at: str
    metadata: dict[str, Any] = {}
