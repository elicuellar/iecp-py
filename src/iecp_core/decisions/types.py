"""Decision Types -- SS19 of the specification.

Decisions are first-class protocol objects that are captured,
tracked, and queryable. They follow a lifecycle:
PROPOSED -> AFFIRMED | REJECTED | SUPERSEDED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from ..types.entity import EntityId
from ..types.event import ConversationId, EventId

# -- Configuration -------------------------------------------------------------


@dataclass(frozen=True)
class DecisionManagerConfig:
    """Configuration for the DecisionManager."""

    require_human_affirmation: bool = True
    """Whether decisions require human affirmation (default: True)."""


DEFAULT_DECISION_MANAGER_CONFIG = DecisionManagerConfig()

# -- Decision Status -----------------------------------------------------------

DecisionStatus = Literal["proposed", "affirmed", "rejected", "superseded"]

# -- Decision ------------------------------------------------------------------


@dataclass
class Decision:
    """A tracked decision within a conversation."""

    event_id: EventId
    conversation_id: ConversationId
    summary: str
    proposed_by: EntityId
    proposed_at: float
    affirmed_by: list[EntityId] = field(default_factory=list)
    rejected_by: list[EntityId] = field(default_factory=list)
    context_events: list[EventId] = field(default_factory=list)
    status: DecisionStatus = "proposed"
    superseded_by: Optional[EventId] = None
