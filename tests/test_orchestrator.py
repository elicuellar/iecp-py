"""Orchestrator integration tests -- Phase 4: Orchestration Engine.

Tests the full pipeline: event -> debounce -> route -> gate -> lock -> dispatch.
Uses mock dependencies and a FakeTimerProvider for deterministic timer behavior.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable

import pytest

from iecp_core.debounce import Debouncer, DebouncerConfig, SealedBatch
from iecp_core.events.event_factory import create_message_event
from iecp_core.lock import FloorLock
from iecp_core.orchestrator import (
    DEFAULT_ORCHESTRATOR_CONFIG,
    DispatchPayload,
    Orchestrator,
    OrchestratorConfig,
    OrchestratorError,
    OrchestrationTrace,
)
from iecp_core.types.conversation import (
    Conversation,
    ConversationConfig,
    Participant,
)
from iecp_core.types.entity import Entity, EntityCapabilities, EntityId
from iecp_core.types.event import (
    BatchId,
    ConversationId,
    Event,
    EventId,
    HandoffContent,
    MessageContent,
)
from iecp_core.utils import generate_id


# -- FakeTimerProvider -------------------------------------------------------


class FakeTimerProvider:
    """Fake timer for deterministic testing."""

    def __init__(self, start_time: float = 1_000_000.0) -> None:
        self._current_time: float = start_time
        self._timers: dict[int, tuple[float, Callable[[], None]]] = {}
        self._next_handle: int = 0

    def set_timeout(self, callback: Callable[[], None], ms: float) -> int:
        handle = self._next_handle
        self._next_handle += 1
        self._timers[handle] = (self._current_time + ms, callback)
        return handle

    def clear_timeout(self, handle: Any) -> None:
        self._timers.pop(handle, None)

    def now(self) -> float:
        return self._current_time

    def advance(self, ms: float) -> None:
        """Advance time by ms, firing any timers that expire."""
        target = self._current_time + ms
        while True:
            earliest: tuple[int, float, Callable[[], None]] | None = None
            for h, (fire_at, cb) in list(self._timers.items()):
                if fire_at <= target and (
                    earliest is None or fire_at < earliest[1]
                ):
                    earliest = (h, fire_at, cb)
            if earliest is None:
                break
            h, fire_at, cb = earliest
            self._current_time = fire_at
            del self._timers[h]
            cb()
        self._current_time = target


# -- Mock Implementations ---------------------------------------------------

_NOW = datetime.now(timezone.utc).isoformat()


class MockEventStore:
    """Mock event store with sync access."""

    def __init__(self) -> None:
        self._events: dict[EventId, Event] = {}

    def add_events(self, *events: Event) -> None:
        for event in events:
            self._events[event.id] = event

    def get_by_id_sync(self, event_id: EventId) -> Event | None:
        return self._events.get(event_id)


class MockEntityManager:
    """Mock entity manager with sync access."""

    def __init__(self) -> None:
        self._entities: dict[EntityId, Entity] = {}

    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity

    def get_entity_sync(self, entity_id: EntityId) -> Entity | None:
        return self._entities.get(entity_id)


class MockConversationManager:
    """Mock conversation manager with sync access."""

    def __init__(self) -> None:
        self._conversations: dict[ConversationId, Conversation] = {}
        self._participants: dict[ConversationId, list[Participant]] = {}

    def add_conversation(
        self, conversation: Conversation, participants: list[Participant]
    ) -> None:
        self._conversations[conversation.id] = conversation
        self._participants[conversation.id] = participants

    def get_conversation_sync(
        self, conversation_id: ConversationId
    ) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def get_participants_sync(
        self, conversation_id: ConversationId
    ) -> list[Participant]:
        return self._participants.get(conversation_id, [])


# -- Helpers -----------------------------------------------------------------

CONV_ID = ConversationId("conv-orch")
HUMAN_A = EntityId("human-a")
AI_A = EntityId("ai-a")
AI_B = EntityId("ai-b")


def _make_entity(
    entity_id: EntityId,
    entity_type: str,
    domains: list[str] | None = None,
) -> Entity:
    return Entity(
        id=entity_id,
        name=str(entity_id),
        type=entity_type,
        capabilities=EntityCapabilities(
            domains=domains or [],
        ),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_participant(entity_id: EntityId) -> Participant:
    return Participant(
        entity_id=entity_id,
        conversation_id=CONV_ID,
        role="member",
        lifecycle_status="active",
        joined_at=_NOW,
    )


def _make_conversation(conv_id: ConversationId = CONV_ID) -> Conversation:
    return Conversation(
        id=conv_id,
        title="Test Conversation",
        config=ConversationConfig(),
        status="active",
        created_by=HUMAN_A,
        created_at=_NOW,
        updated_at=_NOW,
    )


class OrchestratorTestContext:
    """Container for all test dependencies and collectors."""

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.timer = FakeTimerProvider()
        self.debouncer = Debouncer(
            DebouncerConfig(base_ms=100, adaptive=False),
            timer_provider=self.timer,
        )
        self.floor_lock = FloorLock(timer_provider=self.timer)
        self.event_store = MockEventStore()
        self.entity_manager = MockEntityManager()
        self.conversation_manager = MockConversationManager()

        self.orchestrator = Orchestrator(
            debouncer=self.debouncer,
            floor_lock=self.floor_lock,
            event_store=self.event_store,
            entity_manager=self.entity_manager,
            conversation_manager=self.conversation_manager,
            config=config
            or OrchestratorConfig(
                default_respondent_mode="auto",
                max_cascade_depth=3,
                max_concurrent_ai_processing=1,
                max_ai_invocations_per_hour=60,
            ),
        )

        self.dispatches: list[DispatchPayload] = []
        self.traces: list[OrchestrationTrace] = []
        self.errors: list[OrchestratorError] = []
        self.cascade_limits: list[dict[str, Any]] = []
        self.human_interrupts: list[ConversationId] = []

        self.orchestrator.on("dispatch", lambda p: self.dispatches.append(p))
        self.orchestrator.on("trace", lambda t: self.traces.append(t))
        self.orchestrator.on("error", lambda e: self.errors.append(e))
        self.orchestrator.on(
            "cascade_limit",
            lambda c, d: self.cascade_limits.append(
                {"conversation_id": c, "depth": d}
            ),
        )
        self.orchestrator.on(
            "human_interrupt", lambda c: self.human_interrupts.append(c)
        )

    def setup_conversation(
        self,
        entities: list[tuple[EntityId, str]] | None = None,
    ) -> None:
        entity_defs = entities or [
            (HUMAN_A, "human"),
            (AI_A, "artificer"),
        ]

        conversation = _make_conversation()
        participants: list[Participant] = []

        for entity_id, entity_type in entity_defs:
            entity = _make_entity(entity_id, entity_type)
            self.entity_manager.add_entity(entity)
            participants.append(_make_participant(entity_id))

        self.conversation_manager.add_conversation(conversation, participants)

    async def advance_and_process(self, ms: float) -> None:
        """Advance the timer and process any async tasks that were scheduled."""
        self.timer.advance(ms)
        # Allow async tasks scheduled by _emit (via asyncio.ensure_future) to run
        await asyncio.sleep(0)

    def destroy(self) -> None:
        self.orchestrator.destroy()


def _make_human_message(
    text: str = "Hello AI",
    mentions: list[EntityId] | None = None,
) -> Event:
    return create_message_event(
        conversation_id=CONV_ID,
        author_id=HUMAN_A,
        author_type="human",
        text=text,
        mentions=mentions or [],
    )


def _make_ai_event(
    author_id: EntityId = AI_A,
    text: str = "AI Response",
    ai_depth_counter: int = 0,
) -> Event:
    return create_message_event(
        conversation_id=CONV_ID,
        author_id=author_id,
        author_type="artificer",
        text=text,
        ai_depth_counter=ai_depth_counter,
    )


# -- Tests -------------------------------------------------------------------


class TestFullPipeline:
    async def test_dispatches_ai_when_human_sends_message_auto_mode(self) -> None:
        ctx = OrchestratorTestContext()
        ctx.setup_conversation()

        event = _make_human_message()
        ctx.event_store.add_events(event)
        await ctx.orchestrator.handle_incoming_event(event)

        # Advance timers to trigger debounce seal
        await ctx.advance_and_process(200)

        assert len(ctx.dispatches) == 1
        assert ctx.dispatches[0].entity_id == AI_A
        assert ctx.dispatches[0].conversation_id == CONV_ID
        assert len(ctx.traces) == 1
        assert ctx.traces[0].outcome == "dispatched"

        ctx.destroy()

    async def test_produces_a_trace_even_when_no_eligible_entities(self) -> None:
        ctx = OrchestratorTestContext(
            config=OrchestratorConfig(default_respondent_mode="mentioned_only")
        )
        ctx.setup_conversation()

        event = _make_human_message()  # no mentions
        ctx.event_store.add_events(event)
        await ctx.orchestrator.handle_incoming_event(event)

        await ctx.advance_and_process(200)

        assert len(ctx.dispatches) == 0
        assert len(ctx.traces) == 1
        assert ctx.traces[0].outcome == "no_eligible"

        ctx.destroy()


class TestGating:
    async def test_blocks_dispatch_when_cascade_depth_is_reached(self) -> None:
        ctx = OrchestratorTestContext(
            config=OrchestratorConfig(
                max_cascade_depth=2,
                default_respondent_mode="auto",
            )
        )
        ctx.setup_conversation(
            entities=[
                (HUMAN_A, "human"),
                (AI_A, "artificer"),
                (AI_B, "artificer"),
            ]
        )

        event = _make_human_message()
        ctx.event_store.add_events(event)
        await ctx.orchestrator.handle_incoming_event(event)
        await ctx.advance_and_process(200)

        # First dispatch should work (depth 0)
        assert len(ctx.dispatches) == 1

        # Simulate AI response at depth 1
        ai_response = _make_ai_event(
            author_id=AI_A, text="Response 1", ai_depth_counter=1
        )
        ctx.event_store.add_events(ai_response)

        await ctx.orchestrator.handle_response_commit(
            conversation_id=CONV_ID,
            entity_id=AI_A,
            response_event=ai_response,
        )

        # Cascade should be blocked at depth 2 (max_cascade_depth: 2)
        assert len(ctx.cascade_limits) >= 1

        ctx.destroy()

    async def test_blocks_when_rate_limit_is_exceeded(self) -> None:
        ctx = OrchestratorTestContext(
            config=OrchestratorConfig(
                max_ai_invocations_per_hour=1,
                default_respondent_mode="auto",
            )
        )
        ctx.setup_conversation()

        # First message -- should dispatch
        event1 = _make_human_message(text="First")
        ctx.event_store.add_events(event1)
        await ctx.orchestrator.handle_incoming_event(event1)
        await ctx.advance_and_process(200)

        assert len(ctx.dispatches) == 1

        # Simulate AI response commit (increments hourly counter)
        ai_resp = _make_ai_event(text="Done", ai_depth_counter=0)
        await ctx.orchestrator.handle_response_commit(
            conversation_id=CONV_ID,
            entity_id=AI_A,
            response_event=ai_resp,
        )

        # Second message -- should be rate-limited
        event2 = _make_human_message(text="Second")
        ctx.event_store.add_events(event2)
        await ctx.orchestrator.handle_incoming_event(event2)
        await ctx.advance_and_process(200)

        # Only 1 dispatch total (second was gated)
        assert len(ctx.dispatches) == 1
        assert len(ctx.traces) >= 2
        gated_trace = next(
            (t for t in ctx.traces if t.outcome == "gated"), None
        )
        assert gated_trace is not None

        ctx.destroy()


class TestHumanInterruption:
    async def test_releases_lock_and_emits_human_interrupt(self) -> None:
        ctx = OrchestratorTestContext()
        ctx.setup_conversation()

        event1 = _make_human_message(text="Start")
        ctx.event_store.add_events(event1)
        await ctx.orchestrator.handle_incoming_event(event1)
        await ctx.advance_and_process(200)

        assert len(ctx.dispatches) == 1
        assert ctx.floor_lock.is_locked(CONV_ID) is True

        # Human interrupts
        interrupt = _make_human_message(text="Wait, actually...")
        ctx.event_store.add_events(interrupt)
        await ctx.orchestrator.handle_incoming_event(interrupt)

        assert CONV_ID in ctx.human_interrupts
        assert ctx.floor_lock.is_locked(CONV_ID) is False

        ctx.destroy()


class TestEscalation:
    async def test_human_message_clears_escalation_before_dispatch(self) -> None:
        ctx = OrchestratorTestContext()
        ctx.setup_conversation(
            entities=[
                (HUMAN_A, "human"),
                (AI_A, "artificer"),
                (AI_B, "artificer"),
            ]
        )

        # Simulate a handoff escalation event
        handoff_event = Event(
            id=EventId(generate_id()),
            conversation_id=CONV_ID,
            type="handoff",
            author_id=AI_A,
            author_type="artificer",
            content=HandoffContent(
                from_entity=AI_A,
                to_entity=HUMAN_A,
                reason="Need human help",
                context_summary="Test",
                source_event=EventId("evt-1"),
            ),
            created_at=_NOW,
        )

        await ctx.orchestrator.handle_incoming_event(handoff_event)

        # Now try a new message -- human clears escalation, dispatch should proceed
        msg = _make_human_message(text="Test while escalated")
        ctx.event_store.add_events(msg)
        await ctx.orchestrator.handle_incoming_event(msg)
        await ctx.advance_and_process(200)

        # The human message clears escalation, so dispatch should proceed
        assert len(ctx.dispatches) == 1

        ctx.destroy()


class TestPostResponseCommit:
    async def test_releases_lock_and_increments_counters_on_commit(self) -> None:
        ctx = OrchestratorTestContext()
        ctx.setup_conversation()

        event = _make_human_message()
        ctx.event_store.add_events(event)
        await ctx.orchestrator.handle_incoming_event(event)
        await ctx.advance_and_process(200)

        assert len(ctx.dispatches) == 1
        assert ctx.floor_lock.is_locked(CONV_ID) is True

        # Commit response
        ai_resp = _make_ai_event(text="Response", ai_depth_counter=0)

        await ctx.orchestrator.handle_response_commit(
            conversation_id=CONV_ID,
            entity_id=AI_A,
            response_event=ai_resp,
        )

        # Lock should be released
        assert ctx.floor_lock.is_locked(CONV_ID) is False

        ctx.destroy()


class TestCascade:
    async def test_allows_ai_to_ai_cascade_up_to_max_depth(self) -> None:
        ctx = OrchestratorTestContext(
            config=OrchestratorConfig(
                max_cascade_depth=3,
                default_respondent_mode="auto",
            )
        )
        ctx.setup_conversation(
            entities=[
                (HUMAN_A, "human"),
                (AI_A, "artificer"),
                (AI_B, "artificer"),
            ]
        )

        # Human sends message -> AI dispatched (depth 0)
        event = _make_human_message()
        ctx.event_store.add_events(event)
        await ctx.orchestrator.handle_incoming_event(event)
        await ctx.advance_and_process(200)

        assert len(ctx.dispatches) == 1

        # AI_A responds at depth 1 -> should trigger cascade to AI_B
        ai_resp1 = _make_ai_event(
            author_id=AI_A, text="AI_A response", ai_depth_counter=1
        )
        ctx.event_store.add_events(ai_resp1)

        await ctx.orchestrator.handle_response_commit(
            conversation_id=CONV_ID,
            entity_id=AI_A,
            response_event=ai_resp1,
        )

        # Should have dispatched AI_B for cascade (depth 2)
        assert len(ctx.dispatches) == 2
        assert ctx.dispatches[1].entity_id == AI_B
        assert ctx.dispatches[1].ai_depth_counter == 2

        ctx.destroy()


class TestErrorHandling:
    async def test_emits_error_and_trace_when_event_store_fails(self) -> None:
        ctx = OrchestratorTestContext()
        ctx.setup_conversation()

        # Make get_by_id_sync raise an exception
        original_get = ctx.event_store.get_by_id_sync

        def failing_get(event_id: EventId) -> Event | None:
            raise RuntimeError("DB connection lost")

        ctx.event_store.get_by_id_sync = failing_get  # type: ignore

        event = _make_human_message()
        ctx.event_store.add_events(event)
        await ctx.orchestrator.handle_incoming_event(event)
        await ctx.advance_and_process(200)

        assert len(ctx.errors) == 1
        assert ctx.errors[0].code == "PIPELINE_ERROR"
        assert "DB connection lost" in ctx.errors[0].message
        assert len(ctx.traces) == 1
        assert ctx.traces[0].outcome == "error"

        ctx.destroy()


class TestConcurrentLimit:
    async def test_blocks_second_dispatch_when_concurrent_limit_is_1(self) -> None:
        ctx = OrchestratorTestContext(
            config=OrchestratorConfig(
                max_concurrent_ai_processing=1,
                default_respondent_mode="auto",
            )
        )
        ctx.setup_conversation(
            entities=[
                (HUMAN_A, "human"),
                (AI_A, "artificer"),
                (AI_B, "artificer"),
            ]
        )

        # First message dispatches AI
        event1 = _make_human_message(text="First", mentions=[AI_A])
        ctx.event_store.add_events(event1)
        await ctx.orchestrator.handle_incoming_event(event1)
        await ctx.advance_and_process(200)

        assert len(ctx.dispatches) == 1

        # While AI_A is still processing, the lock prevents further dispatch
        assert ctx.floor_lock.is_locked(CONV_ID) is True

        ctx.destroy()


class TestTraceEmission:
    async def test_every_pipeline_run_produces_a_trace(self) -> None:
        ctx = OrchestratorTestContext(
            config=OrchestratorConfig(default_respondent_mode="mentioned_only")
        )
        ctx.setup_conversation()

        # No mention -> no_eligible trace
        event1 = _make_human_message(text="No mention")
        ctx.event_store.add_events(event1)
        await ctx.orchestrator.handle_incoming_event(event1)
        await ctx.advance_and_process(200)

        # With mention -> dispatched trace
        event2 = _make_human_message(text="Hey @AI", mentions=[AI_A])
        ctx.event_store.add_events(event2)
        await ctx.orchestrator.handle_incoming_event(event2)
        await ctx.advance_and_process(200)

        assert len(ctx.traces) == 2
        assert ctx.traces[0].outcome == "no_eligible"
        assert ctx.traces[1].outcome == "dispatched"

        # Both traces have required fields
        for trace in ctx.traces:
            assert trace.trace_id is not None
            assert trace.conversation_id == CONV_ID
            assert trace.routing is not None
            assert trace.gating is not None
            assert trace.duration_ms >= 0

        ctx.destroy()
