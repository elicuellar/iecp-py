"""Orchestrator -- The brain of IECP.

Coordinates the pipeline: event -> debounce -> route -> gate -> lock -> dispatch.
Receives all dependencies via constructor (DI). Manages internal state for
cascade tracking, rate limiting, escalation, and round-robin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..debounce.debouncer import Debouncer
from ..debounce.types import SealedBatch
from ..lock.floor_lock import FloorLock
from ..lock.types import LockRequest, LockResult, LockState
from ..types.conversation import Conversation, Participant
from ..types.entity import Entity, EntityId
from ..types.event import (
    BatchId,
    ConversationId,
    Event,
    EventId,
    HandoffContent,
    MessageContent,
)
from ..utils import generate_id
from .gating import GatingParams, evaluate_gating
from .routing import RoutingParams, resolve_routing
from .types import (
    DEFAULT_ORCHESTRATOR_CONFIG,
    DispatchPayload,
    GatingResult,
    OrchestratorConfig,
    OrchestratorError,
    OrchestrationTrace,
    RoutingDecision,
)


# -- Sync Protocols ----------------------------------------------------------
# The orchestrator pipeline runs synchronously (triggered by debouncer
# callbacks). These protocols define the sync interface the orchestrator
# needs from its dependencies.


class SyncEventStore(Protocol):
    """Synchronous event store interface for the orchestrator."""

    def get_by_id_sync(self, event_id: EventId) -> Event | None: ...


class SyncEntityManager(Protocol):
    """Synchronous entity manager interface for the orchestrator."""

    def get_entity_sync(self, entity_id: EntityId) -> Entity | None: ...


class SyncConversationManager(Protocol):
    """Synchronous conversation manager interface for the orchestrator."""

    def get_conversation_sync(
        self, conversation_id: ConversationId
    ) -> Conversation | None: ...

    def get_participants_sync(
        self, conversation_id: ConversationId
    ) -> list[Participant]: ...


# -- Internal State ----------------------------------------------------------


@dataclass
class _ConversationState:
    """Per-conversation internal state."""

    hourly_invocation_count: int = 0
    hourly_reset_at: float = 0.0
    concurrent_processing_count: int = 0
    active_handoff: HandoffContent | None = None
    escalation_active: bool = False
    last_ai_depth_counter: int = 0
    processing_entities: set[EntityId] = field(default_factory=set)


# -- Orchestrator ------------------------------------------------------------


class Orchestrator:
    """The Orchestrator coordinates the pipeline:
    event -> debounce -> route -> gate -> lock -> dispatch.
    """

    def __init__(
        self,
        debouncer: Debouncer,
        floor_lock: FloorLock,
        event_store: SyncEventStore,
        entity_manager: SyncEntityManager,
        conversation_manager: SyncConversationManager,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._config: OrchestratorConfig = config or DEFAULT_ORCHESTRATOR_CONFIG
        self._debouncer = debouncer
        self._floor_lock = floor_lock
        self._event_store = event_store
        self._entity_manager = entity_manager
        self._conversation_manager = conversation_manager

        self._conversation_state: dict[ConversationId, _ConversationState] = {}
        self._last_served: dict[EntityId, float] = {}
        self._listeners: dict[str, set[Callable[..., Any]]] = {}
        self._destroyed = False

        # Subscribe to debouncer batch_sealed events
        self._debouncer.on("batch_sealed", self._on_batch_sealed)

    # -- Event Emitter -------------------------------------------------------

    def on(self, event: str, listener: Callable[..., Any]) -> None:
        """Register a listener for an orchestrator event."""
        if event not in self._listeners:
            self._listeners[event] = set()
        self._listeners[event].add(listener)

    def off(self, event: str, listener: Callable[..., Any]) -> None:
        """Remove a listener for an orchestrator event."""
        listeners = self._listeners.get(event)
        if listeners:
            listeners.discard(listener)

    def _emit(self, event: str, *args: Any) -> None:
        """Emit an event to all registered listeners."""
        listeners = self._listeners.get(event)
        if not listeners:
            return
        for listener in set(listeners):
            listener(*args)

    # -- State Management ----------------------------------------------------

    def _get_or_create_state(
        self, conversation_id: ConversationId
    ) -> _ConversationState:
        state = self._conversation_state.get(conversation_id)
        if state is None:
            state = _ConversationState(
                hourly_reset_at=time.time() * 1000.0,
            )
            self._conversation_state[conversation_id] = state
        return state

    # -- Public API ----------------------------------------------------------

    def handle_incoming_event(self, event: Event) -> None:
        """Handle an incoming event from the event log.

        Routes human events to the debouncer and handles
        human interruption of AI locks.
        """
        if self._destroyed:
            return

        # Handle handoff events
        if event.type == "handoff":
            content = event.content
            if isinstance(content, HandoffContent):
                state = self._get_or_create_state(event.conversation_id)
                state.active_handoff = content
                state.escalation_active = True
            return

        # Human events
        if event.author_type == "human":
            state = self._get_or_create_state(event.conversation_id)

            # Human responding clears escalation
            if state.escalation_active:
                state.escalation_active = False

            # Human responding clears handoff
            if state.active_handoff and event.type == "message":
                state.active_handoff = None

            # Human resets cascade depth
            state.last_ai_depth_counter = 0

            # Check if an AI holds the lock -> human interruption
            if (
                self._floor_lock.is_locked(event.conversation_id)
                and event.type == "message"
            ):
                self._floor_lock.handle_human_interrupt(event.conversation_id)
                self._emit("human_interrupt", event.conversation_id)

            # Pass to debouncer
            self._debouncer.handle_event(event)
            return

    def handle_typing_start(
        self, conversation_id: ConversationId, author_id: EntityId
    ) -> None:
        """Handle a typing_start signal from a human."""
        if self._destroyed:
            return
        self._debouncer.handle_typing_start(conversation_id, author_id)

    def handle_response_commit(
        self,
        conversation_id: ConversationId,
        entity_id: EntityId,
        response_event: Event,
    ) -> None:
        """Handle an AI response commit.

        Called externally when an AI finishes responding.
        """
        if self._destroyed:
            return

        state = self._get_or_create_state(conversation_id)

        # 1. Release the floor lock
        self._floor_lock.release(conversation_id, entity_id, "commit")

        # 2. Increment hourly invocation counter
        state.hourly_invocation_count += 1

        # 3. Decrement concurrent processing counter
        state.concurrent_processing_count = max(
            0, state.concurrent_processing_count - 1
        )
        state.processing_entities.discard(entity_id)

        # 4. Update last-served for round-robin
        self._last_served[entity_id] = time.time() * 1000.0

        # 5. Track cascade depth
        state.last_ai_depth_counter = response_event.ai_depth_counter

        # 6. Evaluate cascade
        if response_event.ai_depth_counter + 1 < self._config.max_cascade_depth:
            self._run_pipeline_for_ai_response(response_event)
        else:
            self._emit(
                "cascade_limit",
                conversation_id,
                response_event.ai_depth_counter + 1,
            )

    def destroy(self) -> None:
        """Destroy the orchestrator -- clean up all timers and listeners."""
        self._destroyed = True
        self._debouncer.off("batch_sealed", self._on_batch_sealed)
        self._conversation_state.clear()
        self._last_served.clear()
        self._listeners.clear()

    # -- Pipeline ------------------------------------------------------------

    def _on_batch_sealed(self, batch: SealedBatch) -> None:
        """Batch sealed callback -- triggers the orchestration pipeline."""
        if self._destroyed:
            return
        self._run_pipeline(batch)

    def _run_pipeline(self, batch: SealedBatch) -> None:
        """Run the full orchestration pipeline for a sealed batch."""
        start_time = time.time() * 1000.0
        trace_id = generate_id()
        state = self._get_or_create_state(batch.conversation_id)

        # Check hourly reset
        now_ms = time.time() * 1000.0
        if now_ms - state.hourly_reset_at >= 3_600_000:
            state.hourly_invocation_count = 0
            state.hourly_reset_at = now_ms

        routing: RoutingDecision | None = None
        gating: GatingResult | None = None
        lock_result: LockResult | None = None

        try:
            # 1. Load context
            conversation = self._conversation_manager.get_conversation_sync(
                batch.conversation_id
            )
            participants = self._conversation_manager.get_participants_sync(
                batch.conversation_id
            )

            events_result: list[Event] = []
            for event_id in batch.event_ids:
                evt = self._event_store.get_by_id_sync(event_id)
                if evt:
                    events_result.append(evt)

            if conversation is None:
                self._emit_error(
                    "CONVERSATION_NOT_FOUND",
                    f"Conversation {batch.conversation_id} not found",
                    batch.conversation_id,
                )
                self._emit_trace(
                    trace_id,
                    batch,
                    start_time,
                    "error",
                    RoutingDecision(
                        eligible_entities=[],
                        selected_entity=None,
                        reason="Conversation not found",
                        rule_applied="no_eligible",
                    ),
                    GatingResult(allowed=False, checks=[]),
                )
                return

            # Build entity map from participants
            entities: dict[EntityId, Entity] = {}
            for p in participants:
                entity = self._entity_manager.get_entity_sync(p.entity_id)
                if entity:
                    entities[p.entity_id] = entity

            # 2. Resolve routing
            routing = resolve_routing(
                RoutingParams(
                    batch=batch,
                    events=events_result,
                    conversation=conversation,
                    participants=participants,
                    entities=entities,
                    config=self._config,
                    active_handoff=state.active_handoff,
                    last_served=self._last_served,
                )
            )

            # No eligible entity
            if routing.selected_entity is None:
                self._emit_trace(
                    trace_id,
                    batch,
                    start_time,
                    "no_eligible",
                    routing,
                    GatingResult(allowed=False, checks=[]),
                )
                return

            # 3. Evaluate gating
            selected_participant = next(
                (p for p in participants if p.entity_id == routing.selected_entity),
                None,
            )
            entity_status = (
                selected_participant.lifecycle_status
                if selected_participant
                else "left"
            )

            gating = evaluate_gating(
                GatingParams(
                    entity_id=routing.selected_entity,
                    conversation_id=batch.conversation_id,
                    ai_depth_counter=state.last_ai_depth_counter,
                    config=self._config,
                    hourly_invocation_count=state.hourly_invocation_count,
                    concurrent_processing_count=state.concurrent_processing_count,
                    entity_status=entity_status,
                    escalation_active=state.escalation_active,
                )
            )

            if not gating.allowed:
                self._emit_trace(
                    trace_id, batch, start_time, "gated", routing, gating
                )
                cascade_check = next(
                    (c for c in gating.checks if c.name == "cascade_depth"), None
                )
                if cascade_check and not cascade_check.passed:
                    self._emit(
                        "cascade_limit",
                        batch.conversation_id,
                        state.last_ai_depth_counter,
                    )
                return

            # 4. Acquire lock
            lock_result = self._floor_lock.acquire(
                LockRequest(
                    entity_id=routing.selected_entity,
                    conversation_id=batch.conversation_id,
                    estimated_ms=30_000,
                    priority=(
                        "mention"
                        if routing.rule_applied == "explicit_mention"
                        else "default"
                    ),
                )
            )

            if not lock_result.granted:
                self._emit_trace(
                    trace_id,
                    batch,
                    start_time,
                    "gated",
                    routing,
                    gating,
                    lock_result,
                )
                return

            # 5. Dispatch
            state.concurrent_processing_count += 1
            state.processing_entities.add(routing.selected_entity)

            payload = DispatchPayload(
                conversation_id=batch.conversation_id,
                entity_id=routing.selected_entity,
                batch=batch,
                lock=lock_result.lock,
                ai_depth_counter=state.last_ai_depth_counter,
                trace_id=trace_id,
            )

            self._emit("dispatch", payload)

            # 6. Record trace
            self._emit_trace(
                trace_id,
                batch,
                start_time,
                "dispatched",
                routing,
                gating,
                lock_result,
                routing.selected_entity,
            )
        except Exception as err:
            message = str(err)
            self._emit_error("PIPELINE_ERROR", message, batch.conversation_id)
            self._emit_trace(
                trace_id,
                batch,
                start_time,
                "error",
                routing
                or RoutingDecision(
                    eligible_entities=[],
                    selected_entity=None,
                    reason=message,
                    rule_applied="no_eligible",
                ),
                gating or GatingResult(allowed=False, checks=[]),
                lock_result,
            )

    def _run_pipeline_for_ai_response(self, response_event: Event) -> None:
        """Run the pipeline for an AI response (cascade evaluation).
        AI responses skip debouncing -- they trigger directly.
        """
        start_time = time.time() * 1000.0
        trace_id = generate_id()
        conversation_id = response_event.conversation_id
        state = self._get_or_create_state(conversation_id)

        # Create a synthetic batch from the AI response
        synthetic_batch = SealedBatch(
            batch_id=response_event.batch_id or BatchId(generate_id()),
            conversation_id=conversation_id,
            author_id=response_event.author_id,
            event_ids=[response_event.id],
            sealed_at=time.time() * 1000.0,
            message_count=1,
        )

        routing: RoutingDecision | None = None
        gating: GatingResult | None = None
        lock_result: LockResult | None = None

        try:
            conversation = self._conversation_manager.get_conversation_sync(
                conversation_id
            )
            participants = self._conversation_manager.get_participants_sync(
                conversation_id
            )

            if conversation is None:
                return

            entities: dict[EntityId, Entity] = {}
            for p in participants:
                entity = self._entity_manager.get_entity_sync(p.entity_id)
                if entity:
                    entities[p.entity_id] = entity

            # Route using the response event as the batch content
            routing = resolve_routing(
                RoutingParams(
                    batch=synthetic_batch,
                    events=[response_event],
                    conversation=conversation,
                    participants=participants,
                    entities=entities,
                    config=self._config,
                    active_handoff=state.active_handoff,
                    last_served=self._last_served,
                )
            )

            if routing.selected_entity is None:
                self._emit_trace(
                    trace_id,
                    synthetic_batch,
                    start_time,
                    "no_eligible",
                    routing,
                    GatingResult(allowed=False, checks=[]),
                )
                return

            # Gating with incremented depth
            cascade_depth = response_event.ai_depth_counter + 1
            selected_participant = next(
                (p for p in participants if p.entity_id == routing.selected_entity),
                None,
            )
            entity_status = (
                selected_participant.lifecycle_status
                if selected_participant
                else "left"
            )

            gating = evaluate_gating(
                GatingParams(
                    entity_id=routing.selected_entity,
                    conversation_id=conversation_id,
                    ai_depth_counter=cascade_depth,
                    config=self._config,
                    hourly_invocation_count=state.hourly_invocation_count,
                    concurrent_processing_count=state.concurrent_processing_count,
                    entity_status=entity_status,
                    escalation_active=state.escalation_active,
                )
            )

            if not gating.allowed:
                self._emit_trace(
                    trace_id,
                    synthetic_batch,
                    start_time,
                    "gated",
                    routing,
                    gating,
                )
                cascade_check = next(
                    (c for c in gating.checks if c.name == "cascade_depth"), None
                )
                if cascade_check and not cascade_check.passed:
                    self._emit("cascade_limit", conversation_id, cascade_depth)
                return

            # Acquire lock
            lock_result = self._floor_lock.acquire(
                LockRequest(
                    entity_id=routing.selected_entity,
                    conversation_id=conversation_id,
                    estimated_ms=30_000,
                    priority="default",
                )
            )

            if not lock_result.granted:
                self._emit_trace(
                    trace_id,
                    synthetic_batch,
                    start_time,
                    "gated",
                    routing,
                    gating,
                    lock_result,
                )
                return

            state.concurrent_processing_count += 1
            state.processing_entities.add(routing.selected_entity)

            payload = DispatchPayload(
                conversation_id=conversation_id,
                entity_id=routing.selected_entity,
                batch=synthetic_batch,
                lock=lock_result.lock,
                ai_depth_counter=cascade_depth,
                trace_id=trace_id,
            )

            self._emit("dispatch", payload)
            self._emit_trace(
                trace_id,
                synthetic_batch,
                start_time,
                "dispatched",
                routing,
                gating,
                lock_result,
                routing.selected_entity,
            )
        except Exception as err:
            message = str(err)
            self._emit_error("CASCADE_ERROR", message, conversation_id)

    # -- Helpers -------------------------------------------------------------

    def _emit_error(
        self,
        code: str,
        message: str,
        conversation_id: ConversationId | None = None,
        entity_id: EntityId | None = None,
    ) -> None:
        error = OrchestratorError(
            code=code,
            message=message,
            conversation_id=conversation_id,
            entity_id=entity_id,
        )
        self._emit("error", error)

    def _emit_trace(
        self,
        trace_id: str,
        batch: SealedBatch,
        start_time: float,
        outcome: str,
        routing: RoutingDecision,
        gating: GatingResult,
        lock_result: LockResult | None = None,
        dispatch_entity: EntityId | None = None,
    ) -> None:
        trace = OrchestrationTrace(
            trace_id=trace_id,
            conversation_id=batch.conversation_id,
            batch_id=batch.batch_id,
            timestamp=start_time,
            routing=routing,
            gating=gating,
            lock_result=lock_result,
            dispatch_entity=dispatch_entity,
            outcome=outcome,
            duration_ms=(time.time() * 1000.0) - start_time,
        )
        self._emit("trace", trace)
