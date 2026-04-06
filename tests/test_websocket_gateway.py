"""WebSocketGateway tests -- Phase 7.

Uses mock WebSocket objects -- does NOT create a real server.
Tests message routing logic, broadcast, buffering, heartbeat.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

from iecp_core.debounce.types import SealedBatch
from iecp_core.gateway import (
    ConnectionManager,
    DaemonBuffer,
    GatewayClient,
    SimpleTokenValidator,
    WebSocketGateway,
)
from iecp_core.gateway.types import ActiveSignal
from iecp_core.lock.types import LockState
from iecp_core.orchestrator.types import DispatchPayload
from iecp_core.types.entity import EntityId
from iecp_core.types.event import (
    BatchId,
    ConversationId,
    Event,
    EventId,
    MessageContent,
)

# WebSocket readyState constants (mirror the ws library)
WS_OPEN = 1
WS_CLOSED = 3


# -- Mock Helpers ------------------------------------------------------------


class MockWs:
    """Minimal mock WebSocket that records sent messages."""

    def __init__(self, ready_state: int = WS_OPEN) -> None:
        self.readyState = ready_state
        self._sent: list[str] = []
        self.close_calls: list[tuple[int, str]] = []
        self.ping_calls: int = 0

    def send(self, data: str) -> None:
        self._sent.append(data)

    def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_calls.append((code, reason))

    def ping(self) -> None:
        self.ping_calls += 1

    def on(self, *args: object) -> None:
        pass

    def once(self, *args: object) -> None:
        pass

    def remove_listener(self, *args: object) -> None:
        pass

    @property
    def sent_count(self) -> int:
        return len(self._sent)

    def get_messages(self) -> list[dict]:
        return [json.loads(m) for m in self._sent]


def mock_ws(ready_state: int = WS_OPEN) -> MockWs:
    return MockWs(ready_state)


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


def make_event(event_id: str, conv_id: str = "conv-1") -> Event:
    return Event(
        id=EventId(event_id),
        conversation_id=ConversationId(conv_id),
        author_id=EntityId("author-1"),
        author_type="human",
        type="message",
        content=MessageContent(text=f"Message {event_id}", format="plain", mentions=[]),
        is_continuation=False,
        is_complete=True,
        ai_depth_counter=0,
        status="active",
        metadata={},
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def make_batch(conv_id: str = "conv-1") -> SealedBatch:
    return SealedBatch(
        batch_id=BatchId("batch-1"),
        conversation_id=ConversationId(conv_id),
        author_id=EntityId("author-1"),
        event_ids=[EventId("e1")],
        sealed_at=time.time() * 1000.0,
        message_count=1,
    )


def make_dispatch(entity_id: str, conv_id: str = "conv-1") -> DispatchPayload:
    now = time.time() * 1000.0
    return DispatchPayload(
        conversation_id=ConversationId(conv_id),
        entity_id=EntityId(entity_id),
        batch=make_batch(conv_id),
        lock=LockState(
            conversation_id=ConversationId(conv_id),
            holder_id=EntityId(entity_id),
            acquired_at=now,
            ttl_ms=30000,
            estimated_ms=20000,
            expires_at=now + 30000,
            metadata={},
        ),
        ai_depth_counter=1,
        trace_id="trace-1",
    )


def make_lock_state(conv_id: str = "conv-1") -> LockState:
    now = time.time() * 1000.0
    return LockState(
        conversation_id=ConversationId(conv_id),
        holder_id=EntityId("entity-1"),
        acquired_at=now,
        ttl_ms=30000,
        estimated_ms=20000,
        expires_at=now + 30000,
        metadata={},
    )


def make_signal(conv_id: str = "conv-1") -> ActiveSignal:
    now = time.time() * 1000.0
    return ActiveSignal(
        entity_id=EntityId("entity-1"),
        conversation_id=ConversationId(conv_id),
        signal_type="thinking",
        created_at=now,
        expires_at=now + 300000,
    )


# -- Fixture -----------------------------------------------------------------


@pytest.fixture
def gateway_setup() -> tuple[ConnectionManager, DaemonBuffer, SimpleTokenValidator, WebSocketGateway]:
    conn_manager = ConnectionManager()
    daemon_buffer = DaemonBuffer(ttl_ms=5000, max_events=100)
    token_validator = SimpleTokenValidator()
    from iecp_core.gateway.types import GatewayConfig
    gateway = WebSocketGateway(
        connection_manager=conn_manager,
        daemon_buffer=daemon_buffer,
        token_validator=token_validator,
        config=GatewayConfig(heartbeat_interval_ms=30000, heartbeat_timeout_ms=10000),
    )
    return conn_manager, daemon_buffer, token_validator, gateway


# -- Tests -------------------------------------------------------------------


class TestBroadcast:
    def test_sends_to_all_subscribers_of_a_conversation(
        self, gateway_setup: tuple
    ) -> None:
        conn, buf, tv, gw = gateway_setup
        ws1 = mock_ws()
        ws2 = mock_ws()
        c1 = create_client(id="c1", entity_id="e1", ws=ws1)
        c2 = create_client(id="c2", entity_id="e2", ws=ws2)

        conn.add_client(c1)
        conn.add_client(c2)
        conn.subscribe("c1", [ConversationId("conv-1")])
        conn.subscribe("c2", [ConversationId("conv-1")])

        gw.broadcast(ConversationId("conv-1"), {"type": "pong"})

        assert ws1.sent_count == 1
        assert ws2.sent_count == 1

    def test_excludes_specified_client(self, gateway_setup: tuple) -> None:
        conn, buf, tv, gw = gateway_setup
        ws1 = mock_ws()
        ws2 = mock_ws()
        c1 = create_client(id="c1", entity_id="e1", ws=ws1)
        c2 = create_client(id="c2", entity_id="e2", ws=ws2)

        conn.add_client(c1)
        conn.add_client(c2)
        conn.subscribe("c1", [ConversationId("conv-1")])
        conn.subscribe("c2", [ConversationId("conv-1")])

        gw.broadcast(ConversationId("conv-1"), {"type": "pong"}, exclude_client_id="c1")

        assert ws1.sent_count == 0
        assert ws2.sent_count == 1

    def test_does_not_send_to_clients_with_closed_websocket(
        self, gateway_setup: tuple
    ) -> None:
        conn, buf, tv, gw = gateway_setup
        ws1 = mock_ws(ready_state=WS_CLOSED)
        c1 = create_client(id="c1", ws=ws1)
        conn.add_client(c1)
        conn.subscribe("c1", [ConversationId("conv-1")])

        gw.broadcast(ConversationId("conv-1"), {"type": "pong"})
        assert ws1.sent_count == 0


class TestSendToEntity:
    def test_delivers_to_the_correct_client(self, gateway_setup: tuple) -> None:
        conn, buf, tv, gw = gateway_setup
        ws1 = mock_ws()
        c1 = create_client(id="c1", entity_id="e1", ws=ws1)
        conn.add_client(c1)

        result = gw.send_to_entity(EntityId("e1"), {"type": "pong"})
        assert result is True
        assert ws1.sent_count == 1

    def test_returns_false_if_entity_not_connected(self, gateway_setup: tuple) -> None:
        _, _, _, gw = gateway_setup
        result = gw.send_to_entity(EntityId("unknown"), {"type": "pong"})
        assert result is False

    def test_returns_false_if_websocket_is_not_open(
        self, gateway_setup: tuple
    ) -> None:
        conn, buf, tv, gw = gateway_setup
        ws1 = mock_ws(ready_state=WS_CLOSED)
        c1 = create_client(id="c1", entity_id="e1", ws=ws1)
        conn.add_client(c1)

        result = gw.send_to_entity(EntityId("e1"), {"type": "pong"})
        assert result is False


class TestHandleEvent:
    def test_broadcasts_event_to_conversation_subscribers(
        self, gateway_setup: tuple
    ) -> None:
        conn, buf, tv, gw = gateway_setup
        ws1 = mock_ws()
        c1 = create_client(id="c1", ws=ws1)
        conn.add_client(c1)
        conn.subscribe("c1", [ConversationId("conv-1")])

        event = make_event("e1", "conv-1")
        gw.handle_event(event)

        messages = ws1.get_messages()
        assert len(messages) == 1
        assert messages[0]["type"] == "event"


class TestHandleBatchSealed:
    def test_broadcasts_batch_to_subscribers(self, gateway_setup: tuple) -> None:
        conn, buf, tv, gw = gateway_setup
        ws1 = mock_ws()
        c1 = create_client(id="c1", ws=ws1)
        conn.add_client(c1)
        conn.subscribe("c1", [ConversationId("conv-1")])

        gw.handle_batch_sealed(make_batch("conv-1"))

        messages = ws1.get_messages()
        assert len(messages) == 1
        assert messages[0]["type"] == "batch_sealed"


class TestHandleDispatch:
    def test_sends_dispatch_to_target_daemon(self, gateway_setup: tuple) -> None:
        conn, buf, tv, gw = gateway_setup
        ws1 = mock_ws()
        c1 = create_client(id="c1", entity_id="daemon-1", type="daemon", ws=ws1)
        conn.add_client(c1)

        dispatch = make_dispatch("daemon-1")
        gw.handle_dispatch(dispatch)

        messages = ws1.get_messages()
        assert len(messages) == 1
        assert messages[0]["type"] == "dispatch"

    def test_buffers_dispatch_if_daemon_is_disconnected(
        self, gateway_setup: tuple
    ) -> None:
        conn, buf, tv, gw = gateway_setup
        dispatch = make_dispatch("daemon-offline")
        gw.handle_dispatch(dispatch)

        assert buf.has_events(EntityId("daemon-offline"))
        assert buf.get_buffer_size(EntityId("daemon-offline")) == 1


class TestHandleLockState:
    def test_broadcasts_lock_state_to_conversation_subscribers(
        self, gateway_setup: tuple
    ) -> None:
        conn, buf, tv, gw = gateway_setup
        ws1 = mock_ws()
        c1 = create_client(id="c1", ws=ws1)
        conn.add_client(c1)
        conn.subscribe("c1", [ConversationId("conv-1")])

        gw.handle_lock_state(make_lock_state("conv-1"))

        messages = ws1.get_messages()
        assert len(messages) == 1
        assert messages[0]["type"] == "lock_state"


class TestHandleSignal:
    def test_broadcasts_signal_to_conversation_subscribers(
        self, gateway_setup: tuple
    ) -> None:
        conn, buf, tv, gw = gateway_setup
        ws1 = mock_ws()
        c1 = create_client(id="c1", ws=ws1)
        conn.add_client(c1)
        conn.subscribe("c1", [ConversationId("conv-1")])

        gw.handle_signal(make_signal("conv-1"))

        messages = ws1.get_messages()
        assert len(messages) == 1
        assert messages[0]["type"] == "signal"


class TestHandleStreamChunk:
    def test_broadcasts_stream_chunks_to_human_clients_only(
        self, gateway_setup: tuple
    ) -> None:
        conn, buf, tv, gw = gateway_setup
        human_ws = mock_ws()
        daemon_ws = mock_ws()
        human = create_client(id="c1", type="human", entity_id="h1", ws=human_ws)
        daemon = create_client(id="c2", type="daemon", entity_id="d1", ws=daemon_ws)

        conn.add_client(human)
        conn.add_client(daemon)
        conn.subscribe("c1", [ConversationId("conv-1")])
        conn.subscribe("c2", [ConversationId("conv-1")])

        gw.handle_stream_chunk(
            ConversationId("conv-1"),
            EntityId("d1"),
            "Hello ",
            0,
        )

        assert human_ws.sent_count == 1
        assert daemon_ws.sent_count == 0

        messages = human_ws.get_messages()
        assert messages[0]["type"] == "stream_chunk"
        assert messages[0]["payload"]["text"] == "Hello "
        assert messages[0]["payload"]["chunkIndex"] == 0
