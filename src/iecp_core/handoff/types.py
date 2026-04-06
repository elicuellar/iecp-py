"""Handoff & Escalation Types -- SS20 of the specification.

Handoffs transfer conversational responsibility between entities.
Escalations flag that human judgment is required, suppressing AI dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ..types.entity import EntityId
from ..types.event import ConversationId, EventId

# -- Configuration -------------------------------------------------------------


@dataclass(frozen=True)
class HandoffManagerConfig:
    """Configuration for the HandoffManager."""

    max_chain_depth: int = 3
    """Maximum handoff chain depth (default: 3)."""

    handoff_expiry_ms: int = 300_000
    """Handoff expiry in milliseconds (default: 5 minutes)."""


DEFAULT_HANDOFF_MANAGER_CONFIG = HandoffManagerConfig()

# -- Escalation Requires ------------------------------------------------------

EscalationRequires = Literal["approval", "decision", "clarification", "review"]

# -- Active Handoff ------------------------------------------------------------


@dataclass
class ActiveHandoff:
    """An active handoff between entities."""

    event_id: EventId
    conversation_id: ConversationId
    from_entity: EntityId
    to_entity: EntityId
    reason: str
    context_summary: str
    source_event: EventId
    created_at: float
    expires_at: float
    chain_depth: int


# -- Active Escalation --------------------------------------------------------


@dataclass
class ActiveEscalation:
    """An active escalation requiring human judgment."""

    event_id: EventId
    conversation_id: ConversationId
    entity_id: EntityId
    reason: str
    requires: EscalationRequires
    context_summary: str
    source_event: EventId
    created_at: float
    resolved: bool = False
    resolved_by: Optional[EntityId] = None
