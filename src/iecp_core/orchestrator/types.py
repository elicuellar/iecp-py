"""Orchestrator Types -- Phase 4 of the IECP protocol.

Defines configuration, routing decisions, gating results,
dispatch payloads, and orchestration traces for the engine
that coordinates AI dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from ..debounce.types import SealedBatch
from ..lock.types import LockResult, LockState
from ..types.conversation import RespondentMode
from ..types.entity import EntityId
from ..types.event import BatchId, ConversationId, EventId

# -- Configuration -----------------------------------------------------------


@dataclass(frozen=True)
class OrchestratorConfig:
    """Orchestrator-level configuration."""

    max_cascade_depth: int = 3
    """Maximum AI->AI cascade depth before blocking."""

    max_concurrent_ai_processing: int = 1
    """Maximum concurrent AI entities processing per conversation."""

    max_ai_invocations_per_hour: int = 60
    """Maximum AI invocations per hour per conversation."""

    allow_unsolicited_ai: bool = False
    """Allow AIs to speak without being addressed."""

    default_respondent_mode: RespondentMode = "mentioned_only"
    """Default respondent mode."""


DEFAULT_ORCHESTRATOR_CONFIG = OrchestratorConfig()

# -- Routing -----------------------------------------------------------------

RoutingRule = Literal[
    "explicit_mention",
    "mentioned_only_mode",
    "auto_domain_match",
    "auto_round_robin",
    "single_respondent",
    "handoff_override",
    "no_eligible",
    "gated",
    "suppressed",
]


class RoutingDecision(BaseModel):
    """Result of the routing decision."""

    eligible_entities: list[EntityId]
    """All entities that could respond."""

    selected_entity: EntityId | None
    """Entity chosen to respond (None = no dispatch)."""

    reason: str
    """Human-readable explanation."""

    rule_applied: RoutingRule
    """The routing rule that was applied."""


# -- Gating ------------------------------------------------------------------


class GatingCheck(BaseModel):
    """A single gating check result."""

    name: str
    """Name of the check."""

    passed: bool
    """Whether the check passed."""

    detail: str | None = None
    """Optional detail about the check."""


class GatingResult(BaseModel):
    """Aggregate gating result."""

    allowed: bool
    """Whether all checks passed."""

    reason: str | None = None
    """Reason for denial (if denied)."""

    checks: list[GatingCheck]
    """Individual check results."""


# -- Trace -------------------------------------------------------------------


class OrchestrationTrace(BaseModel):
    """A trace of a single orchestration pipeline run."""

    trace_id: str
    """Unique trace identifier (ULID)."""

    conversation_id: ConversationId
    """The conversation this trace belongs to."""

    batch_id: BatchId | None = None
    """The batch that triggered this pipeline run (if any)."""

    trigger_event_id: EventId | None = None
    """The event that triggered this pipeline run (if any)."""

    timestamp: float
    """When the pipeline started."""

    routing: RoutingDecision
    """Routing decision."""

    gating: GatingResult
    """Gating result."""

    lock_result: LockResult | None = None
    """Lock acquisition result (if attempted)."""

    dispatch_entity: EntityId | None = None
    """Entity that was dispatched to (if any)."""

    outcome: Literal["dispatched", "gated", "no_eligible", "suppressed", "error"]
    """Final outcome of the pipeline."""

    duration_ms: float
    """Total pipeline duration in milliseconds."""


# -- Events ------------------------------------------------------------------


class DispatchPayload(BaseModel):
    """Payload emitted when an AI entity should be dispatched."""

    conversation_id: ConversationId
    """The conversation to respond in."""

    entity_id: EntityId
    """The AI entity to dispatch to."""

    batch: SealedBatch
    """The sealed batch to respond to."""

    lock: LockState
    """The lock state after acquisition."""

    ai_depth_counter: int
    """Current cascade depth counter."""

    trace_id: str
    """Trace ID for correlation."""


class OrchestratorError(BaseModel):
    """Error payload emitted by the orchestrator."""

    code: str
    """Error code for programmatic handling."""

    message: str
    """Human-readable error message."""

    conversation_id: ConversationId | None = None
    """The conversation where the error occurred (if applicable)."""

    entity_id: EntityId | None = None
    """The entity involved (if applicable)."""
