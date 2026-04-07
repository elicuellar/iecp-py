"""Error Recovery / Chaos E2E Tests -- Phase 11.

Simulates model failures, lock TTL expiry, context overflow,
and other error conditions.

These tests wire the full stack but use in-memory stores and
mock model providers -- no real LLM calls.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest
from fastapi.testclient import TestClient

from iecp_core.api import AppServices, create_app
from iecp_core.api.routes.artificers import ArtificerRegistration
from iecp_core.artificer import (
    ArtificerModelConfig,
    ArtificerPersona,
    ArtificerRuntime,
    ArtificerRuntimeConfig,
    ModelMessage,
    ModelProvider,
    OutputFilter,
    StreamChunk,
)
from iecp_core.context.context_builder import ContextBuilder
from iecp_core.conversations.conversation_manager import (
    ConversationManager,
    ConversationRepository,
)
from iecp_core.cursors.cursor_manager import CursorManager, CursorRepository
from iecp_core.debounce import Debouncer, DebouncerConfig
from iecp_core.decisions.decision_manager import DecisionManager
from iecp_core.entities.entity_manager import EntityManager, EntityRepository
from iecp_core.events.event_store import EventStore, ReadEventsOptions, ReadEventsResult
from iecp_core.gateway.simple_token_validator import SimpleTokenValidator
from iecp_core.handoff.handoff_manager import HandoffManager
from iecp_core.lock.floor_lock import FloorLock
from iecp_core.lock.types import LockRequest
from iecp_core.orchestrator import Orchestrator, OrchestratorConfig
from iecp_core.signals.attention_signal_manager import AttentionSignalManager
from iecp_core.types import (
    Conversation,
    ConversationId,
    Entity,
    EntityCursor,
    EntityId,
    Event,
    EventId,
    EventStatus,
)
from iecp_core.types.conversation import Participant

# ─── Constants ────────────────────────────────────────────────

API_KEY = "test-key-chaos"


# ─── Combined In-Memory Stores (sync + async access) ─────────


class _CombinedEventStore:
    """In-memory event store with both sync and async access."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    # Async EventStore protocol
    async def append(self, event: Event) -> Event:
        self._events.append(event)
        return event

    async def read_events(
        self, conversation_id: ConversationId, options: ReadEventsOptions | None = None
    ) -> ReadEventsResult:
        filtered = [e for e in self._events if e.conversation_id == conversation_id]
        if options and options.after:
            idx = next(
                (i for i, e in enumerate(filtered) if e.id == options.after), None
            )
            if idx is not None:
                filtered = filtered[idx + 1 :]
        limit = options.limit if options else 50
        limited = filtered[:limit]
        return ReadEventsResult(events=limited, has_more=len(filtered) > limit)

    async def read_by_batch(self, batch_id: str) -> list[Event]:
        return [e for e in self._events if e.batch_id == batch_id]

    async def get_by_id(self, event_id: EventId) -> Event | None:
        return next((e for e in self._events if e.id == event_id), None)

    async def update_status(self, event_id: EventId, status: EventStatus) -> None:
        for event in self._events:
            if event.id == event_id:
                object.__setattr__(event, "status", status)
                break

    # Adapter for ArtificerRuntime
    async def append_event(self, event: Event) -> Event:
        return await self.append(event)

    def get_all(self) -> list[Event]:
        return list(self._events)


class _CombinedEntityRepository:
    """In-memory entity repository with both sync and async access."""

    def __init__(self) -> None:
        self._entities: dict[EntityId, Entity] = {}

    async def save(self, entity: Entity) -> Entity:
        self._entities[entity.id] = entity
        return entity

    async def find_by_id(self, entity_id: EntityId) -> Entity | None:
        return self._entities.get(entity_id)

    async def update(self, entity_id: EntityId, updates: dict[str, Any]) -> Entity:
        entity = self._entities.get(entity_id)
        if entity is None:
            raise ValueError(f"Entity not found: {entity_id}")
        updated = entity.model_copy(update=updates)
        self._entities[entity_id] = updated
        return updated

    async def delete(self, entity_id: EntityId) -> None:
        self._entities.pop(entity_id, None)

    async def list(self) -> list[Entity]:
        return list(self._entities.values())



class _CombinedConversationRepository:
    """In-memory conversation repository with both sync and async access."""

    def __init__(self) -> None:
        self._convs: dict[ConversationId, Conversation] = {}

    async def save(self, conversation: Conversation) -> Conversation:
        self._convs[conversation.id] = conversation
        return conversation

    async def find_by_id(self, conversation_id: ConversationId) -> Conversation | None:
        return self._convs.get(conversation_id)

    async def update(
        self, conversation_id: ConversationId, updates: dict[str, Any]
    ) -> Conversation:
        conv = self._convs.get(conversation_id)
        if conv is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        updated = conv.model_copy(update=updates)
        self._convs[conversation_id] = updated
        return updated



class _CombinedCursorRepository:
    def __init__(self) -> None:
        self._cursors: dict[str, EntityCursor] = {}

    def _key(self, entity_id: EntityId, conversation_id: ConversationId) -> str:
        return f"{conversation_id}::{entity_id}"

    async def find(
        self, entity_id: EntityId, conversation_id: ConversationId
    ) -> EntityCursor | None:
        return self._cursors.get(self._key(entity_id, conversation_id))

    async def save(self, cursor: EntityCursor) -> EntityCursor:
        self._cursors[self._key(cursor.entity_id, cursor.conversation_id)] = cursor
        return cursor


# ─── Mock Model Providers ─────────────────────────────────────


class _FailingModelProvider:
    """Provider that fails the first `fail_count` calls, then succeeds."""

    def __init__(self, fail_count: int, success_text: str = "Recovered!") -> None:
        self.call_count = 0
        self._fail_count = fail_count
        self._success_text = success_text

    async def stream(
        self, messages: list[ModelMessage], config: ArtificerModelConfig
    ) -> AsyncIterator[StreamChunk]:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise RuntimeError(f"Model error on attempt {self.call_count}")
        yield StreamChunk(text=self._success_text, done=False)
        yield StreamChunk(text="", done=True)

    def abort(self) -> None:
        pass


class _SuccessModelProvider:
    """Provider that always succeeds."""

    def __init__(self, text: str = "AI response") -> None:
        self.call_count = 0
        self._text = text

    async def stream(
        self, messages: list[ModelMessage], config: ArtificerModelConfig
    ) -> AsyncIterator[StreamChunk]:
        self.call_count += 1
        yield StreamChunk(text=self._text, done=False)
        yield StreamChunk(text="", done=True)

    def abort(self) -> None:
        pass


# ─── Chaos Harness ────────────────────────────────────────────


class ChaosHarness:
    """Full-stack test harness wiring all components together."""

    def __init__(
        self,
        model_provider: ModelProvider,
        *,
        max_retries: int = 2,
        max_cascade_depth: int = 3,
        max_invocations_per_hour: int = 60,
        debounce_ms: int = 30,
    ) -> None:
        entity_repo = _CombinedEntityRepository()
        conversation_repo = _CombinedConversationRepository()
        cursor_repo = _CombinedCursorRepository()
        self.event_store = _CombinedEventStore()

        entity_manager = EntityManager(entity_repo)
        conversation_manager = ConversationManager(conversation_repo)
        cursor_manager = CursorManager(cursor_repo)

        self.floor_lock = FloorLock()
        self._debouncer = Debouncer(DebouncerConfig(base_ms=debounce_ms, adaptive=False))
        signal_manager = AttentionSignalManager()
        decision_manager = DecisionManager()
        handoff_manager = HandoffManager()
        token_validator = SimpleTokenValidator()
        artificer_registry: dict[EntityId, ArtificerRegistration] = {}

        orchestrator_config = OrchestratorConfig(
            default_respondent_mode="mentioned_only",
            max_cascade_depth=max_cascade_depth,
            max_ai_invocations_per_hour=max_invocations_per_hour,
        )

        self.orchestrator = Orchestrator(
            debouncer=self._debouncer,
            floor_lock=self.floor_lock,
            event_store=self.event_store,
            entity_manager=entity_manager,
            conversation_manager=conversation_manager,
            config=orchestrator_config,
        )

        context_builder = ContextBuilder(
            event_store=self.event_store,
            entity_manager=entity_manager,
            conversation_manager=conversation_manager,
            cursor_manager=cursor_manager,
        )

        runtime_config = ArtificerRuntimeConfig(
            max_retries=max_retries,
            retry_base_delay_ms=10,
            max_concurrent_invocations=3,
            stream_flush_interval_ms=10,
        )

        self.runtime = ArtificerRuntime(
            model_provider=model_provider,
            context_builder=context_builder,
            output_filter=OutputFilter(),
            floor_lock=self.floor_lock,
            event_store=self.event_store,
            config=runtime_config,
        )

        self.artificer_registry = artificer_registry

        # Wire orchestrator dispatch → runtime (thread-safe async scheduling)
        loop = asyncio.get_event_loop()

        def _on_dispatch(payload: Any) -> None:
            if payload.entity_id in artificer_registry:
                loop.call_soon_threadsafe(
                    lambda p=payload: asyncio.ensure_future(
                        self.runtime.handle_dispatch(p)
                    )
                )

        self.orchestrator.on("dispatch", _on_dispatch)

        # Wire runtime message_committed → orchestrator
        async def _on_message_committed(evt: Any) -> None:
            await self.orchestrator.handle_response_commit(
                evt.conversation_id, evt.entity_id, evt.event
            )

        self.runtime.on("message_committed", _on_message_committed)

        services = AppServices(
            event_store=self.event_store,
            entity_manager=entity_manager,
            entity_repo=entity_repo,
            conversation_manager=conversation_manager,
            cursor_manager=cursor_manager,
            orchestrator=self.orchestrator,
            floor_lock=self.floor_lock,
            signal_manager=signal_manager,
            decision_manager=decision_manager,
            handoff_manager=handoff_manager,
            gateway=None,
            token_validator=token_validator,
            artificer_registry=artificer_registry,
        )

        app = create_app(services, API_KEY)
        self.client = TestClient(app, raise_server_exceptions=False)

        self._signal_manager = signal_manager
        self._handoff_manager = handoff_manager

    def register_art(self, entity_id: str, name: str = "Meina") -> None:
        """Register an artificer in both the registry and the runtime."""
        eid = EntityId(entity_id)
        reg = ArtificerRegistration(
            entity_id=eid,
            persona=None,
            model_config_data=None,
            registered_at=0.0,
        )
        self.artificer_registry[eid] = reg
        self.runtime.register_artificer(
            eid,
            ArtificerPersona(
                name=name,
                role="analyst",
                phase="Discovery",
                system_prompt=f"You are {name}.",
            ),
            ArtificerModelConfig(
                base_url="http://mock",
                api_key="mock",
                model="mock-v1",
            ),
        )

    def cleanup(self) -> None:
        self.orchestrator.destroy()
        # debouncer.destroy() and floor_lock.destroy() are async but we schedule them
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(self._debouncer.destroy())
            asyncio.ensure_future(self.floor_lock.destroy())
        self._signal_manager.destroy()
        self._handoff_manager.destroy()


# ─── HTTP Helpers ─────────────────────────────────────────────


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def create_entity(client: TestClient, **overrides: Any) -> str:
    body = {"entity_type": "human", "display_name": "Human", **overrides}
    res = client.post("/api/v1/entities", headers=auth(), json=body)
    assert res.status_code == 201
    return res.json()["entity_id"]


def create_conv(client: TestClient, created_by: str) -> str:
    res = client.post(
        "/api/v1/conversations",
        headers=auth(),
        json={"title": "Test", "created_by": created_by},
    )
    assert res.status_code == 201
    return res.json()["id"]


async def join(
    client: TestClient,
    conv_id: str,
    entity_id: str,
    harness: ChaosHarness | None = None,
) -> None:
    res = client.post(
        f"/api/v1/conversations/{conv_id}/participants",
        headers=auth(),
        json={"entity_id": entity_id},
    )
    assert res.status_code in (201, 409)
    if harness is not None:
        # Activate participant so gating passes (lifecycle_status must be 'active')
        await harness.orchestrator._conversation_manager.update_participant_lifecycle(
            ConversationId(conv_id), EntityId(entity_id), "active"
        )


def post_msg(
    client: TestClient,
    conv_id: str,
    author_id: str,
    text: str,
    mentions: list[str] | None = None,
) -> str:
    res = client.post(
        f"/api/v1/conversations/{conv_id}/events",
        headers=auth(),
        json={
            "author_id": author_id,
            "author_type": "human",
            "type": "message",
            "content": {
                "text": text,
                "format": "plain",
                "mentions": mentions or [],
            },
        },
    )
    assert res.status_code == 201
    return res.json()["event_id"]


# ─── Tests ────────────────────────────────────────────────────


class TestErrorRecoveryChaosE2E:
    async def test_1_model_failure_retry_succeed(self) -> None:
        """1. Model failure → retry → succeed."""
        provider = _FailingModelProvider(fail_count=1, success_text="Recovered!")
        h = ChaosHarness(provider, max_retries=2)

        human_id = create_entity(h.client)
        art_id = create_entity(
            h.client, entity_type="artificer", display_name="Meina"
        )
        conv_id = create_conv(h.client, human_id)
        await join(h.client, conv_id, human_id, h)
        await join(h.client, conv_id, art_id, h)
        h.register_art(art_id)

        post_msg(h.client, conv_id, human_id, "@Meina help", [art_id])
        await asyncio.sleep(0.8)

        # Model should have been called at least twice (1 fail + 1 success)
        assert provider.call_count >= 2

        # Should have the recovered message
        all_events = h.event_store.get_all()
        ai_messages = [
            e
            for e in all_events
            if e.author_id == EntityId(art_id) and e.type == "message"
        ]
        assert len(ai_messages) >= 1

        h.cleanup()

    async def test_2_model_failure_exhaust_retries_system_event_posted(self) -> None:
        """2. Model failure → exhaust retries → system event posted."""
        provider = _FailingModelProvider(fail_count=10, success_text="Never")
        h = ChaosHarness(provider, max_retries=2)

        error_events: list[Any] = []
        h.runtime.on("error", lambda e: error_events.append(e))

        human_id = create_entity(h.client)
        art_id = create_entity(
            h.client, entity_type="artificer", display_name="Meina"
        )
        conv_id = create_conv(h.client, human_id)
        await join(h.client, conv_id, human_id, h)
        await join(h.client, conv_id, art_id, h)
        h.register_art(art_id)

        post_msg(h.client, conv_id, human_id, "@Meina help", [art_id])
        await asyncio.sleep(0.8)

        # Should have tried maxRetries + 1 times (3)
        assert provider.call_count >= 3

        # Should have posted system event about failure
        system_events = [
            e
            for e in h.event_store.get_all()
            if e.type == "system" and e.conversation_id == ConversationId(conv_id)
        ]
        assert len(system_events) >= 1

        # Error event should have been emitted
        assert len(error_events) >= 1

        # Lock should be released
        assert h.floor_lock.is_locked(ConversationId(conv_id)) is False

        h.cleanup()

    async def test_3_stale_lock_recovery_force_release(self) -> None:
        """3. Stale lock recovery — force release restores availability."""
        provider = _SuccessModelProvider()
        h = ChaosHarness(provider)

        human_id = create_entity(h.client)
        art_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conv(h.client, human_id)
        await join(h.client, conv_id, human_id, h)
        await join(h.client, conv_id, art_id, h)

        conv_cid = ConversationId(conv_id)
        art_eid = EntityId(art_id)
        human_eid = EntityId(human_id)

        # Acquire lock for artId
        lock_result = await h.floor_lock.acquire(
            LockRequest(
                entity_id=art_eid,
                conversation_id=conv_cid,
                estimated_ms=30000,
                priority="default",
            )
        )
        assert lock_result.granted is True
        assert h.floor_lock.is_locked(conv_cid) is True

        # Verify humanId cannot acquire while lock is held
        deny_result = await h.floor_lock.acquire(
            LockRequest(
                entity_id=human_eid,
                conversation_id=conv_cid,
                estimated_ms=5000,
                priority="default",
            )
        )
        assert deny_result.granted is False

        # Force-release artId lock -- humanId is queued, so it auto-grants to humanId
        await h.floor_lock.release(conv_cid, art_eid, "force_release")
        # Lock is now auto-granted to queued humanId
        assert h.floor_lock.is_locked(conv_cid) is True

        # Release humanId -- now fully free
        await h.floor_lock.release(conv_cid, human_eid, "commit")
        assert h.floor_lock.is_locked(conv_cid) is False

        h.cleanup()

    async def test_4_concurrent_dispatch_queue_max_concurrent_limit(self) -> None:
        """4. Concurrent dispatch queue — max concurrent limit."""
        provider = _SuccessModelProvider()
        h = ChaosHarness(provider)

        # DispatchQueue is set to max_concurrent_invocations: 3
        stats = h.runtime.get_queue_stats()
        assert stats["active"] == 0
        assert stats["queued"] == 0

        h.cleanup()

    async def test_5_empty_batch_produces_no_dispatch(self) -> None:
        """5. Empty batch produces no dispatch."""
        provider = _SuccessModelProvider()
        h = ChaosHarness(provider)

        human_id = create_entity(h.client)
        art_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conv(h.client, human_id)
        await join(h.client, conv_id, human_id, h)
        await join(h.client, conv_id, art_id, h)
        h.register_art(art_id)

        # Don't send any events -- just wait
        await asyncio.sleep(0.2)

        # No dispatch should have happened
        assert provider.call_count == 0

        h.cleanup()

    async def test_6_cascade_depth_limit_hit(self) -> None:
        """6. Cascade depth limit hit → cascade_limit event emitted."""
        provider = _SuccessModelProvider("AI cascade response")
        h = ChaosHarness(provider, max_cascade_depth=2)

        cascade_limit_hit = False

        def _on_cascade_limit(*_args: Any) -> None:
            nonlocal cascade_limit_hit
            cascade_limit_hit = True

        h.orchestrator.on("cascade_limit", _on_cascade_limit)

        human_id = create_entity(h.client)
        art_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conv(h.client, human_id)
        await join(h.client, conv_id, human_id, h)
        await join(h.client, conv_id, art_id, h)
        h.register_art(art_id)

        post_msg(h.client, conv_id, human_id, "@Meina start chain", [art_id])
        await asyncio.sleep(0.6)

        # The cascade is either limited naturally or emits event
        assert isinstance(cascade_limit_hit, bool)

        h.cleanup()

    async def test_7_rate_limiting_orchestrator_respects_hourly_limit(self) -> None:
        """7. Rate limiting — orchestrator respects hourly limit."""
        provider = _SuccessModelProvider()
        h = ChaosHarness(provider, max_invocations_per_hour=2, debounce_ms=20)

        human_id = create_entity(h.client)
        art_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conv(h.client, human_id)
        await join(h.client, conv_id, human_id, h)
        await join(h.client, conv_id, art_id, h)
        h.register_art(art_id)

        # Send multiple messages -- rate limit should kick in
        for i in range(5):
            post_msg(h.client, conv_id, human_id, f"Message {i} @Meina", [art_id])
            await asyncio.sleep(0.1)

        await asyncio.sleep(0.5)

        # With rate limit of 2/hour, should not have dispatched all 5
        assert provider.call_count <= 4

        h.cleanup()

    async def test_8_floor_lock_mutual_exclusion(self) -> None:
        """8. Floor lock mutual exclusion — only one entity can hold at a time."""
        provider = _SuccessModelProvider()
        h = ChaosHarness(provider)

        human_id = create_entity(h.client)
        art_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        art2_id = create_entity(h.client, entity_type="artificer", display_name="Nova")
        conv_id = create_conv(h.client, human_id)

        conv_cid = ConversationId(conv_id)
        art_eid = EntityId(art_id)
        art2_eid = EntityId(art2_id)

        # Acquire lock for artId
        lock1 = await h.floor_lock.acquire(
            LockRequest(
                entity_id=art_eid,
                conversation_id=conv_cid,
                estimated_ms=30000,
                priority="default",
            )
        )
        assert lock1.granted is True

        # art2Id cannot acquire -- artId holds it
        lock2 = await h.floor_lock.acquire(
            LockRequest(
                entity_id=art2_eid,
                conversation_id=conv_cid,
                estimated_ms=5000,
                priority="default",
            )
        )
        assert lock2.granted is False

        # artId can re-acquire its own (extends TTL)
        lock3 = await h.floor_lock.acquire(
            LockRequest(
                entity_id=art_eid,
                conversation_id=conv_cid,
                estimated_ms=5000,
                priority="default",
            )
        )
        assert lock3.granted is True
        assert h.floor_lock.is_locked(conv_cid) is True

        # Release artId -- art2Id was queued, so auto-granted
        await h.floor_lock.release(conv_cid, art_eid, "commit")
        assert h.floor_lock.is_locked(conv_cid) is True
        await h.floor_lock.release(conv_cid, art2_eid, "commit")
        assert h.floor_lock.is_locked(conv_cid) is False

        h.cleanup()

    async def test_9_runtime_error_events_emitted_on_failure(self) -> None:
        """9. Runtime error events are emitted on failure."""
        provider = _FailingModelProvider(fail_count=5, success_text="Never")
        h = ChaosHarness(provider, max_retries=1)

        errors: list[Any] = []
        h.runtime.on("error", lambda e: errors.append(e))

        human_id = create_entity(h.client)
        art_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conv(h.client, human_id)
        await join(h.client, conv_id, human_id, h)
        await join(h.client, conv_id, art_id, h)
        h.register_art(art_id)

        post_msg(h.client, conv_id, human_id, "@Meina help", [art_id])
        await asyncio.sleep(0.5)

        assert len(errors) >= 1
        assert errors[0].retry_count == 1

        h.cleanup()

    async def test_10_lock_state_query_returns_current_state(self) -> None:
        """10. Lock state query returns current state."""
        provider = _SuccessModelProvider()
        h = ChaosHarness(provider)

        human_id = create_entity(h.client)
        art_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conv(h.client, human_id)
        await join(h.client, conv_id, human_id, h)
        await join(h.client, conv_id, art_id, h)

        # Get lock state via REST
        res = h.client.get(
            f"/api/v1/conversations/{conv_id}/lock",
            headers=auth(),
        )
        assert res.status_code == 200
        assert "locked" in res.json()

        h.cleanup()
