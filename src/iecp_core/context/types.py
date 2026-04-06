"""Context Assembly Types -- Phase 5 of the IECP specification.

Defines the context payload structure, token budget configuration,
participant summaries, and the swappable token estimation interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from ..types.entity import EntityCapabilities, EntityId, EntityLifecycleStatus, EntityType
from ..types.event import BatchId, ConversationId, Event


# -- Participant Summary ------------------------------------------------------


class ParticipantSummary(BaseModel):
    """A compact summary of a participant in a conversation."""

    entity_id: EntityId
    display_name: str
    entity_type: EntityType
    capabilities: EntityCapabilities | None = None
    lifecycle_status: EntityLifecycleStatus


# -- Context Payload ----------------------------------------------------------


class ContextPayload(BaseModel):
    """The assembled context delivered to an AI entity."""

    conversation_id: ConversationId
    recipient_id: EntityId

    unread_messages: list[Event]
    recent_history: list[Event]
    conversation_summary: str | None = None

    participants: list[ParticipantSummary]

    response_expected: bool
    batch_id: BatchId

    your_role: str
    your_capabilities: list[str]
    your_instructions: str | None = None

    active_decisions: list[Event]
    pending_handoffs: list[Event]

    token_budget: int
    tokens_used: int


# -- Configuration ------------------------------------------------------------


@dataclass(frozen=True)
class ContextBuilderConfig:
    """Configuration for the ContextBuilder."""

    default_token_budget: int = 100_000
    """Total token budget for the assembled payload."""

    system_prompt_budget: int = 2_000
    """Token budget reserved for system prompt."""

    participant_budget_per_entity: int = 100
    """Token budget per participant entry."""

    decisions_handoffs_budget: int = 500
    """Token budget reserved for decisions + handoffs."""

    summary_budget: int = 1_000
    """Token budget reserved for conversation summary."""

    recent_history_max_events: int = 50
    """Maximum events to include in recent history."""

    summary_trigger_messages: int = 20
    """Number of messages between summary updates."""


DEFAULT_CONTEXT_BUILDER_CONFIG = ContextBuilderConfig()


# -- Token Estimator ----------------------------------------------------------


class TokenEstimator(Protocol):
    """Interface for estimating token counts."""

    def estimate(self, text: str) -> int:
        """Estimate the token count of a plain text string."""
        ...

    def estimate_event(self, event: Event) -> int:
        """Estimate the token count of an event (serialized)."""
        ...
