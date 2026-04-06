"""ConnectionManager tests -- Phase 7."""

from __future__ import annotations

import time

import pytest

from iecp_core.gateway import ConnectionManager, GatewayClient
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId


# -- Helpers -----------------------------------------------------------------


def mock_ws() -> object:
    """Create a minimal mock WebSocket object."""

    class MockWs:
        readyState = 1  # OPEN

        def send(self, data: str) -> None:
            pass

        def close(self, code: int = 1000, reason: str = "") -> None:
            pass

        def ping(self) -> None:
            pass

        def on(self, *args: object) -> None:
            pass

        def once(self, *args: object) -> None:
            pass

        def remove_listener(self, *args: object) -> None:
            pass

    return MockWs()


def create_client(**overrides: object) -> GatewayClient:
    now = time.time() * 1000.0
    return GatewayClient(
        id=overrides.get("id", "client-1"),  # type: ignore[arg-type]
        type=overrides.get("type", "human"),  # type: ignore[arg-type]
        entity_id=EntityId(overrides.get("entity_id", "entity-1")),  # type: ignore[arg-type]
        conversation_ids=overrides.get("conversation_ids", set()),  # type: ignore[arg-type]
        ws=overrides.get("ws", mock_ws()),
        connected_at=overrides.get("connected_at", now),  # type: ignore[arg-type]
        last_ping_at=overrides.get("last_ping_at", now),  # type: ignore[arg-type]
        authenticated=overrides.get("authenticated", True),  # type: ignore[arg-type]
    )


# -- Tests -------------------------------------------------------------------


class TestAddRemoveClient:
    def test_adds_and_retrieves_a_client_by_id(self) -> None:
        manager = ConnectionManager()
        client = create_client(id="c1")
        manager.add_client(client)
        assert manager.get_client("c1") is client

    def test_removes_a_client(self) -> None:
        manager = ConnectionManager()
        client = create_client(id="c1")
        manager.add_client(client)
        manager.remove_client("c1")
        assert manager.get_client("c1") is None

    def test_removing_nonexistent_client_is_noop(self) -> None:
        manager = ConnectionManager()
        # Should not raise
        manager.remove_client("nonexistent")


class TestGetClientByEntity:
    def test_looks_up_client_by_entity_id(self) -> None:
        manager = ConnectionManager()
        client = create_client(id="c1", entity_id="ent-A")
        manager.add_client(client)
        assert manager.get_client_by_entity(EntityId("ent-A")) is client

    def test_returns_none_for_unknown_entity(self) -> None:
        manager = ConnectionManager()
        assert manager.get_client_by_entity(EntityId("nope")) is None

    def test_cleans_up_entity_index_on_remove(self) -> None:
        manager = ConnectionManager()
        client = create_client(id="c1", entity_id="ent-A")
        manager.add_client(client)
        manager.remove_client("c1")
        assert manager.get_client_by_entity(EntityId("ent-A")) is None


class TestSubscribeUnsubscribe:
    def test_subscribes_a_client_to_conversations(self) -> None:
        manager = ConnectionManager()
        client = create_client(id="c1")
        manager.add_client(client)
        manager.subscribe("c1", [ConversationId("conv-1"), ConversationId("conv-2")])

        assert client in manager.get_subscribers(ConversationId("conv-1"))
        assert client in manager.get_subscribers(ConversationId("conv-2"))

    def test_unsubscribes_a_client_from_conversations(self) -> None:
        manager = ConnectionManager()
        client = create_client(id="c1")
        manager.add_client(client)
        manager.subscribe("c1", [ConversationId("conv-1"), ConversationId("conv-2")])
        manager.unsubscribe("c1", [ConversationId("conv-1")])

        assert client not in manager.get_subscribers(ConversationId("conv-1"))
        assert client in manager.get_subscribers(ConversationId("conv-2"))

    def test_subscribe_is_noop_for_unknown_client(self) -> None:
        manager = ConnectionManager()
        # Should not raise
        manager.subscribe("nope", [ConversationId("conv-1")])


class TestGetSubscribers:
    def test_returns_correct_clients_for_a_conversation(self) -> None:
        manager = ConnectionManager()
        c1 = create_client(id="c1", entity_id="e1")
        c2 = create_client(id="c2", entity_id="e2")
        c3 = create_client(id="c3", entity_id="e3")

        manager.add_client(c1)
        manager.add_client(c2)
        manager.add_client(c3)

        manager.subscribe("c1", [ConversationId("conv-A")])
        manager.subscribe("c2", [ConversationId("conv-A")])
        manager.subscribe("c3", [ConversationId("conv-B")])

        subs_a = manager.get_subscribers(ConversationId("conv-A"))
        assert len(subs_a) == 2
        assert c1 in subs_a
        assert c2 in subs_a
        assert c3 not in subs_a

    def test_returns_empty_list_for_unknown_conversation(self) -> None:
        manager = ConnectionManager()
        assert manager.get_subscribers(ConversationId("nope")) == []

    def test_handles_client_subscribed_via_constructor_conversation_ids(self) -> None:
        manager = ConnectionManager()
        client = create_client(
            id="c1",
            conversation_ids={ConversationId("conv-X")},
        )
        manager.add_client(client)
        assert client in manager.get_subscribers(ConversationId("conv-X"))


class TestGetConnectedDaemons:
    def test_returns_only_daemon_entity_ids(self) -> None:
        manager = ConnectionManager()
        manager.add_client(create_client(id="c1", type="human", entity_id="h1"))
        manager.add_client(create_client(id="c2", type="daemon", entity_id="d1"))
        manager.add_client(create_client(id="c3", type="daemon", entity_id="d2"))

        daemons = manager.get_connected_daemons()
        assert len(daemons) == 2
        assert EntityId("d1") in daemons
        assert EntityId("d2") in daemons


class TestGetAllClients:
    def test_returns_all_connected_clients(self) -> None:
        manager = ConnectionManager()
        manager.add_client(create_client(id="c1", entity_id="e1"))
        manager.add_client(create_client(id="c2", entity_id="e2"))
        assert len(manager.get_all_clients()) == 2


class TestMultipleClientsPerConversation:
    def test_multiple_clients_subscribe_to_same_conversation(self) -> None:
        manager = ConnectionManager()
        clients = [
            create_client(id=f"c{i}", entity_id=f"e{i}") for i in range(5)
        ]
        for c in clients:
            manager.add_client(c)
        for c in clients:
            manager.subscribe(c.id, [ConversationId("conv-shared")])

        assert len(manager.get_subscribers(ConversationId("conv-shared"))) == 5


class TestDestroy:
    def test_clears_all_state(self) -> None:
        manager = ConnectionManager()
        manager.add_client(create_client(id="c1", entity_id="e1"))
        manager.subscribe("c1", [ConversationId("conv-1")])
        manager.destroy()

        assert manager.get_all_clients() == []
        assert manager.get_subscribers(ConversationId("conv-1")) == []
