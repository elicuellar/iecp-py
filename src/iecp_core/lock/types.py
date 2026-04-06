"""Floor Lock Types -- Phase 3 of the IECP protocol.

Defines configuration, lock state, request/result structures, and queue entries
for the mutual exclusion mechanism that prevents multiple AI entities from
responding simultaneously in the same conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from ..types.entity import EntityId
from ..types.event import ConversationId

# -- Configuration -----------------------------------------------------------


@dataclass(frozen=True)
class FloorLockConfig:
    """Configuration for the Floor Lock system. All durations in milliseconds."""

    default_ttl_ms: int = 30_000
    """Default TTL when computed TTL is below this floor."""

    max_ttl_ms: int = 60_000
    """Maximum TTL cap."""

    ttl_multiplier: float = 1.5
    """Multiplier applied to estimated_ms."""

    grace_period_ms: int = 5_000
    """Time after TTL before force-release."""


DEFAULT_FLOOR_LOCK_CONFIG = FloorLockConfig()

# -- Lock Priority -----------------------------------------------------------

LockPriority = Literal["mention", "artificer", "daemon", "default"]

PRIORITY_ORDER: dict[LockPriority, int] = {
    "mention": 3,
    "artificer": 2,
    "daemon": 1,
    "default": 0,
}

# -- Lock State --------------------------------------------------------------


class LockState(BaseModel):
    """The current state of a held lock on a conversation."""

    conversation_id: ConversationId
    holder_id: EntityId
    acquired_at: float
    ttl_ms: float
    estimated_ms: float
    expires_at: float
    metadata: dict[str, Any] = {}


# -- Lock Request ------------------------------------------------------------


class LockRequest(BaseModel):
    """A request to acquire the Floor Lock."""

    entity_id: EntityId
    conversation_id: ConversationId
    estimated_ms: float
    priority: LockPriority = "default"
    metadata: dict[str, Any] | None = None


# -- Lock Result -------------------------------------------------------------


class LockResult(BaseModel):
    """Result of a lock acquisition attempt."""

    granted: bool
    lock: LockState | None = None
    reason: str | None = None
    queue_position: int | None = None


# -- Lock Release ------------------------------------------------------------

LockReleaseReason = Literal[
    "commit", "yield", "ttl_expired", "human_interrupt", "force_release"
]


class LockRelease(BaseModel):
    """Information about a lock release event."""

    conversation_id: ConversationId
    entity_id: EntityId
    reason: LockReleaseReason


# -- Events ------------------------------------------------------------------

# LockAcquiredEvent is just a LockState
LockAcquiredEvent = LockState

# LockReleasedEvent is LockRelease + state
class LockReleasedEvent(BaseModel):
    """Event emitted when a lock is released."""

    conversation_id: ConversationId
    entity_id: EntityId
    reason: LockReleaseReason
    state: LockState


# -- Queue Entry -------------------------------------------------------------


class QueueEntry(BaseModel):
    """An entry in the per-conversation wait queue."""

    entity_id: EntityId
    priority: LockPriority
    queued_at: float
    estimated_ms: float
    metadata: dict[str, Any] = {}
    expires_at: float
