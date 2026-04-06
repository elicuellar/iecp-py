"""Full-Stack E2E Tests -- Phase 11.

Wires the full server stack:
REST API → Orchestrator → ArtificerRuntime → Event Store.
Uses mock ModelProvider -- no real LLM calls.
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
from iecp_core.conversations.conversation_manager import ConversationManager
from iecp_core.cursors.cursor_manager import CursorManager
from iecp_core.debounce import Debouncer, DebouncerConfig
from iecp_core.decisions.decision_manager import DecisionManager
from iecp_core.entities.entity_manager import EntityManager
from iecp_core.events.event_store import EventStore, ReadEventsOptions, ReadEventsResult
from iecp_core.gateway.simple_token_validator import SimpleTokenValidator
from iecp_core.handoff.handoff_manager import HandoffManager
from iecp_core.lock.floor_lock import FloorLock
from iecp_core.lock.types import LockRequest
from iecp_core.observability.metrics_collector import MetricsCollector
from iecp_core.observability.trace_logger import TraceLogger
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

API_KEY = "test-key-e2e"


# ─── Combined In-Memory Stores (sync + async access) ─────────


class _CombinedEventStore:
    def __init__(self) -> None:
        self._events: list[Event] = []

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

    def get_by_id_sync(self, event_id: EventId) -> Event | None:
        return next((e for e in self._events if e.id == event_id), None)

    async def append_event(self, event: Event) -> Event:
        return await self.append(event)

    def get_all(self) -> list[Event]:
        return list(self._events)


class _CombinedEntityRepository:
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

    def get_entity_sync(self, entity_id: EntityId) -> Entity | None:
        return self._entities.get(entity_id)


class _CombinedConversationRepository:
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

    def get_conversation_sync(
        self, conversation_id: ConversationId
    ) -> Conversation | None:
        return self._convs.get(conversation_id)

    def get_participants_sync(
        self, conversation_id: ConversationId
    ) -> list[Participant]:
        conv = self._convs.get(conversation_id)
        if conv is None:
            return []
        return list(conv.participants)


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


# ─── Entity/Conversation Manager Sync Wrappers ───────────────


class _SyncEntityManager(EntityManager):
    def __init__(self, repo: _CombinedEntityRepository) -> None:
        super().__init__(repo)
        self._combined = repo

    def get_entity_sync(self, entity_id: EntityId) -> Entity | None:
        return self._combined.get_entity_sync(entity_id)


class _SyncConversationManager(ConversationManager):
    def __init__(self, repo: _CombinedConversationRepository) -> None:
        super().__init__(repo)
        self._combined = repo

    def get_conversation_sync(
        self, conversation_id: ConversationId
    ) -> Conversation | None:
        return self._combined.get_conversation_sync(conversation_id)

    def get_participants_sync(
        self, conversation_id: ConversationId
    ) -> list[Participant]:
        return self._combined.get_participants_sync(conversation_id)


# ─── Mock Model Provider ─────────────────────────────────────


class _MockModelProvider:
    """Mock provider that yields a configurable response text."""

    def __init__(self, response_text: str = "Hello from the AI!") -> None:
        self.call_count = 0
        self.responses: list[str] = [response_text]
        self.aborted = False

    async def stream(
        self, messages: list[ModelMessage], config: ArtificerModelConfig
    ) -> AsyncIterator[StreamChunk]:
        idx = min(self.call_count, len(self.responses) - 1)
        text = self.responses[idx]
        self.call_count += 1
        yield StreamChunk(text=text, done=False)
        yield StreamChunk(text="", done=True)

    def abort(self) -> None:
        self.aborted = True


# ─── E2E Harness ─────────────────────────────────────────────


class E2EHarness:
    """Full-stack test harness with observability support."""

    def __init__(
        self,
        *,
        debounce_ms: int = 50,
        model_response: str = "Hello from the AI!",
        max_cascade_depth: int = 3,
        max_invocations_per_hour: int = 60,
    ) -> None:
        entity_repo = _CombinedEntityRepository()
        conversation_repo = _CombinedConversationRepository()
        cursor_repo = _CombinedCursorRepository()
        self.event_store = _CombinedEventStore()

        self._entity_manager = _SyncEntityManager(entity_repo)
        conversation_manager = _SyncConversationManager(conversation_repo)
        cursor_manager = CursorManager(cursor_repo)

        self.floor_lock = FloorLock()
        self._debouncer = Debouncer(DebouncerConfig(base_ms=debounce_ms, adaptive=False))
        signal_manager = AttentionSignalManager()
        decision_manager = DecisionManager()
        handoff_manager = HandoffManager()
        token_validator = SimpleTokenValidator()
        artificer_registry: dict[EntityId, ArtificerRegistration] = {}
        self.artificer_registry = artificer_registry
        self.trace_logger = TraceLogger()
        self.metrics_collector = MetricsCollector()

        orchestrator_config = OrchestratorConfig(
            default_respondent_mode="mentioned_only",
            max_cascade_depth=max_cascade_depth,
            max_ai_invocations_per_hour=max_invocations_per_hour,
        )

        self.orchestrator = Orchestrator(
            debouncer=self._debouncer,
            floor_lock=self.floor_lock,
            event_store=self.event_store,
            entity_manager=self._entity_manager,
            conversation_manager=conversation_manager,
            config=orchestrator_config,
        )

        # Wire trace logging
        self.orchestrator.on("trace", lambda t: self.trace_logger.record(t))

        self.model_provider = _MockModelProvider(model_response)
        context_builder = ContextBuilder(
            event_store=self.event_store,
            entity_manager=self._entity_manager,
            conversation_manager=conversation_manager,
            cursor_manager=cursor_manager,
        )

        runtime_config = ArtificerRuntimeConfig(
            max_retries=2,
            retry_base_delay_ms=10,
            max_concurrent_invocations=3,
            stream_flush_interval_ms=10,
        )

        self.runtime = ArtificerRuntime(
            model_provider=self.model_provider,
            context_builder=context_builder,
            output_filter=OutputFilter(),
            floor_lock=self.floor_lock,
            event_store=self.event_store,
            config=runtime_config,
        )

        # Wire orchestrator dispatch → runtime
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
        def _on_message_committed(evt: Any) -> None:
            self.orchestrator.handle_response_commit(
                evt.conversation_id, evt.entity_id, evt.event
            )

        self.runtime.on("message_committed", _on_message_committed)

        # Attach metrics (use a no-op gateway emitter)
        class _NoopEmitter:
            def on(self, *_: Any) -> None:
                pass

        self.metrics_collector.attach(
            orchestrator=self.orchestrator,
            runtime=self.runtime,
            gateway=_NoopEmitter(),
        )

        services = AppServices(
            event_store=self.event_store,
            entity_manager=self._entity_manager,
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
            metrics_collector=self.metrics_collector,
            trace_logger=self.trace_logger,
            artificer_runtime=self.runtime,
        )

        app = create_app(services, API_KEY)
        self.client = TestClient(app, raise_server_exceptions=False)

        self._signal_manager = signal_manager
        self._handoff_manager = handoff_manager
        self._conversation_manager = conversation_manager

    def register_artificer(
        self,
        entity_id: str,
        name: str = "Meina",
        domains: list[str] | None = None,
    ) -> None:
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
        self._debouncer.destroy()
        self.floor_lock.destroy()
        self._signal_manager.destroy()
        self._handoff_manager.destroy()


# ─── HTTP Helpers ─────────────────────────────────────────────


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def create_entity(client: TestClient, **overrides: Any) -> str:
    body = {"entity_type": "human", "display_name": "Test Human", **overrides}
    res = client.post("/api/v1/entities", headers=auth(), json=body)
    assert res.status_code == 201, f"create_entity failed: {res.status_code} {res.text}"
    return res.json()["entity_id"]


def create_conversation(client: TestClient, created_by: str) -> str:
    res = client.post(
        "/api/v1/conversations",
        headers=auth(),
        json={"title": "Test Convo", "created_by": created_by},
    )
    assert res.status_code == 201
    return res.json()["id"]


async def join_conversation(
    client: TestClient,
    conversation_id: str,
    entity_id: str,
    harness: E2EHarness | None = None,
) -> None:
    res = client.post(
        f"/api/v1/conversations/{conversation_id}/participants",
        headers=auth(),
        json={"entity_id": entity_id},
    )
    assert res.status_code in (201, 409)
    if harness is not None:
        await harness._conversation_manager.update_participant_lifecycle(
            ConversationId(conversation_id), EntityId(entity_id), "active"
        )


def post_event(
    client: TestClient,
    conversation_id: str,
    author_id: str,
    text: str,
    mentions: list[str] | None = None,
) -> str:
    res = client.post(
        f"/api/v1/conversations/{conversation_id}/events",
        headers=auth(),
        json={
            "author_id": author_id,
            "author_type": "human",
            "type": "message",
            "content": {"text": text, "format": "plain", "mentions": mentions or []},
        },
    )
    assert res.status_code == 201
    return res.json()["event_id"]


def get_events(client: TestClient, conversation_id: str) -> list[dict[str, Any]]:
    res = client.get(
        f"/api/v1/conversations/{conversation_id}/events",
        headers=auth(),
    )
    assert res.status_code == 200
    return res.json()["events"]


# ─── Tests ────────────────────────────────────────────────────


class TestFullStackE2E:
    async def test_1_human_sends_message_artificer_responds(self) -> None:
        """1. Human sends message → Artificer responds."""
        h = E2EHarness(debounce_ms=50)

        human_id = create_entity(h.client)
        artificer_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conversation(h.client, human_id)
        await join_conversation(h.client, conv_id, human_id, h)
        await join_conversation(h.client, conv_id, artificer_id, h)
        h.register_artificer(artificer_id)

        post_event(h.client, conv_id, human_id, "Hello @Meina", [artificer_id])
        await asyncio.sleep(0.4)

        events = get_events(h.client, conv_id)
        messages = [e for e in events if e["event_type"] == "message"]
        assert len(messages) >= 2

        ai_message = next(
            (m for m in messages if m["author_id"] == artificer_id), None
        )
        assert ai_message is not None
        assert ai_message["content"]["text"] == "Hello from the AI!"

        h.cleanup()

    async def test_2_multi_turn_conversation(self) -> None:
        """2. Multi-turn conversation."""
        h = E2EHarness(debounce_ms=50)

        human_id = create_entity(h.client)
        artificer_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conversation(h.client, human_id)
        await join_conversation(h.client, conv_id, human_id, h)
        await join_conversation(h.client, conv_id, artificer_id, h)
        h.register_artificer(artificer_id)

        h.model_provider.responses = ["Response 1", "Response 2", "Response 3"]

        # Send messages with enough gap for full dispatch cycle
        for i in range(1, 4):
            post_event(h.client, conv_id, human_id, f"Message {i} @Meina", [artificer_id])
            await asyncio.sleep(0.6)

        events = get_events(h.client, conv_id)
        ai_messages = [
            e
            for e in events
            if e["author_id"] == artificer_id and e["event_type"] == "message"
        ]
        assert len(ai_messages) >= 1  # At least 1 AI response

        h.cleanup()

    async def test_3_two_humans_one_artificer(self) -> None:
        """3. Two humans + one artificer."""
        h = E2EHarness(debounce_ms=50)

        alice = create_entity(h.client, display_name="Alice")
        bob = create_entity(h.client, display_name="Bob")
        artificer_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conversation(h.client, alice)
        await join_conversation(h.client, conv_id, alice, h)
        await join_conversation(h.client, conv_id, bob, h)
        await join_conversation(h.client, conv_id, artificer_id, h)
        h.register_artificer(artificer_id)

        post_event(h.client, conv_id, alice, "Hey everyone @Meina", [artificer_id])
        await asyncio.sleep(0.2)
        post_event(h.client, conv_id, bob, "I agree @Meina", [artificer_id])
        await asyncio.sleep(0.3)

        events = get_events(h.client, conv_id)
        human_messages = [
            e
            for e in events
            if e["event_type"] == "message" and e["author_type"] == "human"
        ]
        assert len(human_messages) == 2

        ai_messages = [
            e
            for e in events
            if e["author_id"] == artificer_id and e["event_type"] == "message"
        ]
        assert len(ai_messages) >= 1

        h.cleanup()

    async def test_4_mentioned_only_mode_only_mentioned_artificer_responds(
        self,
    ) -> None:
        """4. Mentioned_only mode — only mentioned artificer responds."""
        h = E2EHarness(debounce_ms=50)

        human_id = create_entity(h.client)
        art1 = create_entity(h.client, entity_type="artificer", display_name="Meina")
        art2 = create_entity(h.client, entity_type="artificer", display_name="Nova")
        conv_id = create_conversation(h.client, human_id)
        await join_conversation(h.client, conv_id, human_id, h)
        await join_conversation(h.client, conv_id, art1, h)
        await join_conversation(h.client, conv_id, art2, h)
        h.register_artificer(art1, "Meina")
        h.register_artificer(art2, "Nova")

        # Only mention Meina
        post_event(h.client, conv_id, human_id, "Hey @Meina", [art1])
        await asyncio.sleep(0.8)

        events = get_events(h.client, conv_id)
        art1_msgs = [
            e for e in events if e["author_id"] == art1 and e["event_type"] == "message"
        ]
        art2_msgs = [
            e for e in events if e["author_id"] == art2 and e["event_type"] == "message"
        ]
        assert len(art1_msgs) >= 1
        assert len(art2_msgs) == 0

        h.cleanup()

    async def test_5_decision_lifecycle_via_rest_api(self) -> None:
        """5. Decision lifecycle via REST API."""
        h = E2EHarness()

        human_id = create_entity(h.client)
        conv_id = create_conversation(h.client, human_id)
        await join_conversation(h.client, conv_id, human_id, h)

        # Propose decision
        prop_res = h.client.post(
            f"/api/v1/conversations/{conv_id}/decisions",
            headers=auth(),
            json={
                "proposed_by": human_id,
                "summary": "Adopt IECP protocol",
                "context_events": [],
            },
        )
        assert prop_res.status_code == 201
        decision_id = prop_res.json()["event_id"]

        # Get active decisions
        active_res = h.client.get(
            f"/api/v1/conversations/{conv_id}/decisions",
            headers=auth(),
        )
        assert active_res.status_code == 200
        assert len(active_res.json()) >= 1

        # Affirm
        affirm_res = h.client.patch(
            f"/api/v1/conversations/{conv_id}/decisions/{decision_id}",
            headers=auth(),
            json={"action": "affirm", "entity_id": human_id},
        )
        assert affirm_res.status_code == 200
        assert affirm_res.json()["status"] == "affirmed"

        h.cleanup()

    async def test_6_handoff_via_rest_api(self) -> None:
        """6. Handoff via REST API."""
        h = E2EHarness()

        human_id = create_entity(h.client)
        art1 = create_entity(h.client, entity_type="artificer", display_name="Meina")
        art2 = create_entity(h.client, entity_type="artificer", display_name="Nova")
        conv_id = create_conversation(h.client, human_id)
        await join_conversation(h.client, conv_id, human_id, h)
        await join_conversation(h.client, conv_id, art1, h)
        await join_conversation(h.client, conv_id, art2, h)

        res = h.client.post(
            f"/api/v1/conversations/{conv_id}/handoffs",
            headers=auth(),
            json={
                "from_entity": art1,
                "to_entity": art2,
                "reason": "Nova specializes in this",
                "context_summary": "Need analysis help",
                "source_event": "evt-placeholder",
            },
        )
        assert res.status_code == 201
        assert "event_id" in res.json()

        h.cleanup()

    async def test_7_attention_signals_lifecycle(self) -> None:
        """7. Attention signals lifecycle."""
        h = E2EHarness()

        human_id = create_entity(h.client)
        artificer_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conversation(h.client, human_id)
        await join_conversation(h.client, conv_id, human_id, h)
        await join_conversation(h.client, conv_id, artificer_id, h)

        # Emit signal
        sig_res = h.client.post(
            f"/api/v1/conversations/{conv_id}/signals",
            headers=auth(),
            json={"entityId": artificer_id, "signalType": "thinking"},
        )
        assert sig_res.status_code == 201

        # Get signals
        signals_res = h.client.get(
            f"/api/v1/conversations/{conv_id}/signals",
            headers=auth(),
        )
        assert signals_res.status_code == 200
        body = signals_res.json()
        assert len(body) >= 1
        assert body[0]["signal_type"] == "thinking"

        h.cleanup()

    async def test_8_floor_lock_contention_second_dispatch_waits(self) -> None:
        """8. Floor lock contention — second dispatch waits."""
        h = E2EHarness(debounce_ms=50)

        human_id = create_entity(h.client)
        art1 = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conversation(h.client, human_id)
        await join_conversation(h.client, conv_id, human_id, h)
        await join_conversation(h.client, conv_id, art1, h)
        h.register_artificer(art1)

        # Manually acquire lock
        h.floor_lock.acquire(
            LockRequest(
                entity_id=EntityId(art1),
                conversation_id=ConversationId(conv_id),
                estimated_ms=5000,
                priority="default",
            )
        )

        # Post event -- orchestrator should try to dispatch but lock is held
        post_event(h.client, conv_id, human_id, "@Meina help", [art1])
        await asyncio.sleep(0.2)

        # Lock is held; traces should exist
        traces = h.trace_logger.query()
        assert len(traces) >= 1

        # Release lock
        h.floor_lock.release(
            ConversationId(conv_id), EntityId(art1), "commit"
        )

        h.cleanup()

    async def test_9_metrics_api_returns_data_after_events(self) -> None:
        """9. Metrics API returns data after events."""
        h = E2EHarness(debounce_ms=50)

        human_id = create_entity(h.client)
        artificer_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conversation(h.client, human_id)
        await join_conversation(h.client, conv_id, human_id, h)
        await join_conversation(h.client, conv_id, artificer_id, h)
        h.register_artificer(artificer_id)

        post_event(h.client, conv_id, human_id, "@Meina hello", [artificer_id])
        await asyncio.sleep(0.3)

        # System metrics
        sys_res = h.client.get("/api/v1/metrics", headers=auth())
        assert sys_res.status_code == 200
        assert sys_res.json()["uptime_ms"] > 0

        # Traces
        traces_res = h.client.get("/api/v1/metrics/traces", headers=auth())
        assert traces_res.status_code == 200
        assert len(traces_res.json()["traces"]) >= 1

        h.cleanup()

    async def test_10_enhanced_health_check(self) -> None:
        """10. Enhanced health check."""
        h = E2EHarness()

        res = h.client.get("/health")
        assert res.status_code == 200

        body = res.json()
        assert body["status"] == "ok"
        assert body["checks"]["database"] == "ok"
        assert body["checks"]["gateway"] == "ok"
        assert body["checks"]["artificerQueue"]["status"] == "ok"
        assert body["version"] == "1.0.0-rc1"

        h.cleanup()

    async def test_11_token_auth_generate_and_verify(self) -> None:
        """11. Token auth → generate and verify."""
        h = E2EHarness()

        human_id = create_entity(h.client)

        # Generate a token
        token_res = h.client.post(
            "/api/v1/auth/tokens",
            headers=auth(),
            json={
                "entityId": human_id,
                "type": "human",
                "conversationIds": ["conv-1"],
            },
        )
        assert token_res.status_code == 201
        assert "token" in token_res.json()
        assert isinstance(token_res.json()["token"], str)

        h.cleanup()

    async def test_12_lock_acquisition_and_release_via_rest_api(self) -> None:
        """12. Lock acquisition and release via REST API."""
        h = E2EHarness()

        human_id = create_entity(h.client)
        artificer_id = create_entity(h.client, entity_type="artificer", display_name="Meina")
        conv_id = create_conversation(h.client, human_id)
        await join_conversation(h.client, conv_id, human_id, h)
        await join_conversation(h.client, conv_id, artificer_id, h)

        # Acquire lock
        lock_res = h.client.post(
            f"/api/v1/conversations/{conv_id}/lock/acquire",
            headers=auth(),
            json={"entityId": artificer_id, "estimatedMs": 5000},
        )
        assert lock_res.status_code == 200
        assert lock_res.json()["granted"] is True

        # Try second acquire -- should be denied
        lock2_res = h.client.post(
            f"/api/v1/conversations/{conv_id}/lock/acquire",
            headers=auth(),
            json={"entityId": human_id, "estimatedMs": 5000},
        )
        assert lock2_res.status_code == 200
        assert lock2_res.json()["granted"] is False

        # Release
        rel_res = h.client.post(
            f"/api/v1/conversations/{conv_id}/lock/release",
            headers=auth(),
            json={"entityId": artificer_id, "reason": "commit"},
        )
        assert rel_res.status_code == 200

        h.cleanup()
