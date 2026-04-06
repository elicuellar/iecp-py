"""Gateway Types -- Phase 7 of the IECP protocol.

Defines WebSocket gateway types for real-time event delivery
to humans and daemon CLIs (§3.2, §10.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from ..debounce.types import SealedBatch
from ..lock.types import LockState
from ..orchestrator.types import DispatchPayload
from ..types.entity import EntityId
from ..types.event import ConversationId, Event

# -- Client Types ------------------------------------------------------------

ClientType = Literal["human", "daemon"]

# -- Gateway Client ----------------------------------------------------------


@dataclass
class GatewayClient:
    """A connected WebSocket client."""

    id: str
    """Unique connection ID."""

    type: ClientType
    """Human or daemon."""

    entity_id: EntityId
    """The entity this client represents."""

    conversation_ids: set[ConversationId]
    """Conversations this client is subscribed to."""

    ws: Any
    """The underlying WebSocket (or mock)."""

    connected_at: float
    """Unix ms when the client connected."""

    last_ping_at: float
    """Unix ms of last received pong/message."""

    authenticated: bool
    """Whether the client has authenticated."""


# -- Server -> Client Messages -----------------------------------------------


class EventMessage(BaseModel):
    type: Literal["event"] = "event"
    payload: Event


class BatchSealedMessage(BaseModel):
    type: Literal["batch_sealed"] = "batch_sealed"
    payload: SealedBatch


class DispatchMessage(BaseModel):
    type: Literal["dispatch"] = "dispatch"
    payload: DispatchPayload


class LockStateMessage(BaseModel):
    type: Literal["lock_state"] = "lock_state"
    payload: LockState


class SignalMessage(BaseModel):
    type: Literal["signal"] = "signal"
    payload: "ActiveSignal"


class TypingPayload(BaseModel):
    conversation_id: ConversationId
    entity_id: EntityId


class TypingMessage(BaseModel):
    type: Literal["typing"] = "typing"
    payload: TypingPayload


class StreamChunkPayload(BaseModel):
    conversation_id: ConversationId
    entity_id: EntityId
    text: str
    chunk_index: int


class StreamChunkMessage(BaseModel):
    type: Literal["stream_chunk"] = "stream_chunk"
    payload: StreamChunkPayload


class ErrorPayload(BaseModel):
    code: str
    message: str


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    payload: ErrorPayload


class ConnectedPayload(BaseModel):
    client_id: str
    entity_id: EntityId


class ConnectedMessage(BaseModel):
    type: Literal["connected"] = "connected"
    payload: ConnectedPayload


class BufferedEventsMessage(BaseModel):
    type: Literal["buffered_events"] = "buffered_events"
    payload: list[Event]


class PongMessage(BaseModel):
    type: Literal["pong"] = "pong"


ServerMessage = (
    EventMessage
    | BatchSealedMessage
    | DispatchMessage
    | LockStateMessage
    | SignalMessage
    | TypingMessage
    | StreamChunkMessage
    | ErrorMessage
    | ConnectedMessage
    | BufferedEventsMessage
    | PongMessage
)

# -- Client -> Server Messages -----------------------------------------------


class SubscribeMessage(BaseModel):
    type: Literal["subscribe"] = "subscribe"
    conversation_ids: list[ConversationId]


class UnsubscribeMessage(BaseModel):
    type: Literal["unsubscribe"] = "unsubscribe"
    conversation_ids: list[ConversationId]


class TypingStartMessage(BaseModel):
    type: Literal["typing_start"] = "typing_start"
    conversation_id: ConversationId


class StreamChunkClientMessage(BaseModel):
    type: Literal["stream_chunk"] = "stream_chunk"
    conversation_id: ConversationId
    text: str


class PingMessage(BaseModel):
    type: Literal["ping"] = "ping"


ClientMessage = (
    SubscribeMessage
    | UnsubscribeMessage
    | TypingStartMessage
    | StreamChunkClientMessage
    | PingMessage
)

# -- Configuration -----------------------------------------------------------


@dataclass(frozen=True)
class GatewayConfig:
    """Gateway configuration."""

    port: int = 8080
    """WebSocket server port."""

    heartbeat_interval_ms: float = 30_000
    """Interval between heartbeat pings (ms)."""

    heartbeat_timeout_ms: float = 10_000
    """Timeout waiting for pong after ping (ms)."""

    daemon_buffer_ttl_ms: float = 300_000
    """How long to buffer events for disconnected daemons (ms)."""

    daemon_buffer_max_events: int = 1000
    """Maximum events to buffer per daemon."""

    max_connections: int = 200
    """Maximum concurrent connections."""


DEFAULT_GATEWAY_CONFIG = GatewayConfig()

# -- Authentication ----------------------------------------------------------


class AuthToken(BaseModel):
    """Authentication token payload."""

    entity_id: EntityId
    """The entity this token authenticates."""

    type: ClientType
    """Client type."""

    conversation_ids: list[ConversationId]
    """Conversations this token grants access to."""


class TokenValidator(Protocol):
    """Token validation interface -- swappable for JWT, API key lookup, etc."""

    async def validate(self, token: str) -> AuthToken | None: ...


# -- ActiveSignal (mirrored from TS core for gateway tests) ------------------

AttentionSignalType = Literal["listening", "thinking", "deferred", "acknowledged"]


class ActiveSignal(BaseModel):
    """An active attention signal."""

    entity_id: EntityId
    conversation_id: ConversationId
    signal_type: AttentionSignalType
    note: str | None = None
    batch_id: str | None = None
    created_at: float
    expires_at: float
