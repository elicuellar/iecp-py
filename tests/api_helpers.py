"""Shared test helpers for API tests -- Phase 10.

Creates a test app with in-memory repositories and returns
a TestClient for making requests.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from iecp_core.api import AppServices, create_app
from iecp_core.api.routes.artificers import ArtificerRegistration
from iecp_core.conversations.conversation_manager import ConversationManager, ConversationRepository
from iecp_core.cursors.cursor_manager import CursorManager, CursorRepository
from iecp_core.decisions.decision_manager import DecisionManager
from iecp_core.debounce import Debouncer
from iecp_core.entities.entity_manager import EntityManager, EntityRepository
from iecp_core.events.event_store import EventStore, ReadEventsOptions, ReadEventsResult
from iecp_core.gateway.simple_token_validator import SimpleTokenValidator
from iecp_core.handoff.handoff_manager import HandoffManager
from iecp_core.lock.floor_lock import FloorLock
from iecp_core.orchestrator import Orchestrator
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

TEST_API_KEY = "test-admin-key-12345"


# --- In-memory stores -------------------------------------------------------


class MockEventStore:
    def __init__(self) -> None:
        self._events: list[Event] = []

    async def append(self, event: Event) -> Event:
        self._events.append(event)
        return event

    async def read_events(
        self,
        conversation_id: ConversationId,
        options: ReadEventsOptions | None = None,
    ) -> ReadEventsResult:
        filtered = [e for e in self._events if e.conversation_id == conversation_id]

        if options and options.after:
            idx = next(
                (i for i, e in enumerate(filtered) if e.id == options.after), None
            )
            if idx is not None:
                filtered = filtered[idx + 1:]

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
                object.__setattr__(event, "status", status) if hasattr(event, "__slots__") else setattr(event, "status", status)
                break


class MockEntityRepository:
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


class MockConversationRepository:
    def __init__(self) -> None:
        self._convs: dict[ConversationId, Conversation] = {}

    async def save(self, conversation: Conversation) -> Conversation:
        self._convs[conversation.id] = conversation
        return conversation

    async def find_by_id(self, conversation_id: ConversationId) -> Conversation | None:
        return self._convs.get(conversation_id)

    async def update(self, conversation_id: ConversationId, updates: dict[str, Any]) -> Conversation:
        conv = self._convs.get(conversation_id)
        if conv is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        updated = conv.model_copy(update=updates)
        self._convs[conversation_id] = updated
        return updated


class MockCursorRepository:
    def __init__(self) -> None:
        self._cursors: dict[str, EntityCursor] = {}

    def _key(self, entity_id: EntityId, conversation_id: ConversationId) -> str:
        return f"{conversation_id}::{entity_id}"

    async def find(self, entity_id: EntityId, conversation_id: ConversationId) -> EntityCursor | None:
        return self._cursors.get(self._key(entity_id, conversation_id))

    async def save(self, cursor: EntityCursor) -> EntityCursor:
        self._cursors[self._key(cursor.entity_id, cursor.conversation_id)] = cursor
        return cursor


# --- Test Context -----------------------------------------------------------


def create_test_app(
    *,
    metrics_collector: Any = None,
    trace_logger: Any = None,
) -> tuple[TestClient, AppServices]:
    entity_repo = MockEntityRepository()
    conversation_repo = MockConversationRepository()
    cursor_repo = MockCursorRepository()
    event_store = MockEventStore()

    entity_manager = EntityManager(entity_repo)
    conversation_manager = ConversationManager(conversation_repo)
    cursor_manager = CursorManager(cursor_repo)

    floor_lock = FloorLock()
    signal_manager = AttentionSignalManager()
    decision_manager = DecisionManager()
    handoff_manager = HandoffManager()
    token_validator = SimpleTokenValidator()
    artificer_registry: dict[EntityId, ArtificerRegistration] = {}

    debouncer = Debouncer()
    orchestrator = Orchestrator(
        debouncer=debouncer,
        floor_lock=floor_lock,
        event_store=event_store,
        entity_manager=entity_manager,
        conversation_manager=conversation_manager,
    )

    services = AppServices(
        event_store=event_store,
        entity_manager=entity_manager,
        entity_repo=entity_repo,
        conversation_manager=conversation_manager,
        cursor_manager=cursor_manager,
        orchestrator=orchestrator,
        floor_lock=floor_lock,
        signal_manager=signal_manager,
        decision_manager=decision_manager,
        handoff_manager=handoff_manager,
        gateway=None,
        token_validator=token_validator,
        artificer_registry=artificer_registry,
        metrics_collector=metrics_collector,
        trace_logger=trace_logger,
    )

    app = create_app(services, TEST_API_KEY)
    client = TestClient(app, raise_server_exceptions=True)
    return client, services


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TEST_API_KEY}"}
