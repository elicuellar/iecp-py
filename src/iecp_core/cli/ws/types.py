"""WebSocket Client Types -- messages exchanged between CLI and IECP server.

Mirrors packages/cli/src/ws/types.ts exactly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from ...types.entity import EntityId, EntityLifecycleStatus
from ...types.event import ConversationId, Event, EventId
from ...context.types import ContextPayload, ParticipantSummary

# ─── Client -> Server Messages ────────────────────────────────────────────────


class AuthenticateMessage(BaseModel):
    type: Literal["authenticate"] = "authenticate"
    token: str
    conversation_id: ConversationId
    display_name: str | None = None


class GetRoomStatusMessage(BaseModel):
    type: Literal["get_room_status"] = "get_room_status"
    request_id: str
    conversation_id: ConversationId


class FetchUnreadBatchMessage(BaseModel):
    type: Literal["fetch_unread_batch"] = "fetch_unread_batch"
    request_id: str
    conversation_id: ConversationId
    entity_id: EntityId


class FetchHistoryMessage(BaseModel):
    type: Literal["fetch_history"] = "fetch_history"
    request_id: str
    conversation_id: ConversationId
    limit: int
    before_id: EventId | None = None


class AcquireLockMessage(BaseModel):
    type: Literal["acquire_speaking_lock"] = "acquire_speaking_lock"
    request_id: str
    conversation_id: ConversationId
    entity_id: EntityId
    estimated_ms: int
    intent_summary: str | None = None


class AppendStreamChunkMessage(BaseModel):
    type: Literal["append_stream_chunk"] = "append_stream_chunk"
    request_id: str
    conversation_id: ConversationId
    entity_id: EntityId
    text: str


class CommitMessageMessage(BaseModel):
    type: Literal["commit_message"] = "commit_message"
    request_id: str
    conversation_id: ConversationId
    entity_id: EntityId
    mentions: list[EntityId] | None = None
    metadata: dict[str, Any] | None = None


class YieldFloorMessage(BaseModel):
    type: Literal["yield_floor"] = "yield_floor"
    request_id: str
    conversation_id: ConversationId
    entity_id: EntityId


class ReportActionMessage(BaseModel):
    type: Literal["report_action"] = "report_action"
    request_id: str
    conversation_id: ConversationId
    entity_id: EntityId
    action_type: str
    description: str
    result: Any | None = None
    status: Literal["initiated", "in_progress", "completed", "failed"]


class SignalAttentionMessage(BaseModel):
    type: Literal["signal_attention"] = "signal_attention"
    request_id: str
    conversation_id: ConversationId
    entity_id: EntityId
    signal: Literal["listening", "thinking", "deferred", "acknowledged"]
    utterance_ref: EventId | None = None
    note: str | None = None


class ProposeDecisionMessage(BaseModel):
    type: Literal["propose_decision"] = "propose_decision"
    request_id: str
    conversation_id: ConversationId
    entity_id: EntityId
    summary: str
    context_events: list[EventId] | None = None


class HandoffToMessage(BaseModel):
    type: Literal["handoff_to"] = "handoff_to"
    request_id: str
    conversation_id: ConversationId
    entity_id: EntityId
    to_entity: EntityId
    reason: str
    context_summary: str
    source_event: EventId | None = None


class DisconnectMessage(BaseModel):
    type: Literal["disconnect"] = "disconnect"


ClientMessage = (
    AuthenticateMessage
    | GetRoomStatusMessage
    | FetchUnreadBatchMessage
    | FetchHistoryMessage
    | AcquireLockMessage
    | AppendStreamChunkMessage
    | CommitMessageMessage
    | YieldFloorMessage
    | ReportActionMessage
    | SignalAttentionMessage
    | ProposeDecisionMessage
    | HandoffToMessage
    | DisconnectMessage
)

# ─── Server -> Client Messages ────────────────────────────────────────────────

DecisionStatus = Literal["proposed", "accepted", "rejected", "superseded"]


class AuthenticatedResponse(BaseModel):
    type: Literal["authenticated"] = "authenticated"
    entity_id: EntityId


class ErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    request_id: str | None = None
    code: str
    message: str


class RoomStatusResponse(BaseModel):
    type: Literal["room_status"] = "room_status"
    request_id: str
    conversation_id: ConversationId
    lock_holder: ParticipantSummary | None
    participants: list[ParticipantSummary]
    ai_depth_counter: int
    your_status: EntityLifecycleStatus


class UnreadBatchResponse(BaseModel):
    type: Literal["unread_batch"] = "unread_batch"
    request_id: str
    payload: ContextPayload


class HistoryResponse(BaseModel):
    type: Literal["history"] = "history"
    request_id: str
    messages: list[Event]
    has_more: bool


class LockAcquiredResponse(BaseModel):
    type: Literal["lock_acquired"] = "lock_acquired"
    request_id: str
    granted: bool
    reason: str | None = None
    ttl_ms: int | None = None


class ChunkAckResponse(BaseModel):
    type: Literal["chunk_ack"] = "chunk_ack"
    request_id: str
    ok: bool


class CommitResponse(BaseModel):
    type: Literal["commit_response"] = "commit_response"
    request_id: str
    event_id: EventId
    created_at: int


class YieldResponse(BaseModel):
    type: Literal["yield_response"] = "yield_response"
    request_id: str
    ok: bool


class ActionResponse(BaseModel):
    type: Literal["action_response"] = "action_response"
    request_id: str
    event_id: EventId


class SignalResponse(BaseModel):
    type: Literal["signal_response"] = "signal_response"
    request_id: str
    ok: bool


class DecisionResponse(BaseModel):
    type: Literal["decision_response"] = "decision_response"
    request_id: str
    event_id: EventId
    decision_status: DecisionStatus


class HandoffResponse(BaseModel):
    type: Literal["handoff_response"] = "handoff_response"
    request_id: str
    event_id: EventId


class NewBatchNotification(BaseModel):
    type: Literal["new_batch"] = "new_batch"
    conversation_id: ConversationId


ServerMessage = (
    AuthenticatedResponse
    | ErrorResponse
    | RoomStatusResponse
    | UnreadBatchResponse
    | HistoryResponse
    | LockAcquiredResponse
    | ChunkAckResponse
    | CommitResponse
    | YieldResponse
    | ActionResponse
    | SignalResponse
    | DecisionResponse
    | HandoffResponse
    | NewBatchNotification
)
