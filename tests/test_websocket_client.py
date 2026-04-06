"""WebSocketClient tests -- Phase 8.

Uses mock WebSocket objects -- does NOT create real connections.
Mirrors packages/cli/tests/ws/websocket-client.test.ts exactly.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

import pytest

from iecp_core.cli.ws.websocket_client import WebSocketClient, WebSocketClientConfig, WS_OPEN, WS_CLOSED


# ─── Mock WebSocket ────────────────────────────────────────────────────────────


class MockWebSocket:
    """Minimal mock WebSocket that emits events and records sent messages."""

    OPEN = WS_OPEN
    CLOSED = WS_CLOSED

    def __init__(self) -> None:
        self.readyState = MockWebSocket.OPEN
        self.sent_messages: list[str] = []
        self._listeners: dict[str, list[Callable[..., Any]]] = {}

    def on(self, event: str, listener: Callable[..., Any]) -> None:
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def emit(self, event: str, *args: Any) -> None:
        for listener in list(self._listeners.get(event, [])):
            listener(*args)

    def send(self, data: str) -> None:
        self.sent_messages.append(data)

    def close(self, code: int = 1000, reason: str = "") -> None:
        self.readyState = MockWebSocket.CLOSED
        self.emit("close", code, reason.encode() if isinstance(reason, str) else reason)

    def terminate(self) -> None:
        self.readyState = MockWebSocket.CLOSED


# Global instance to track the current mock WS
_mock_ws_instance: MockWebSocket | None = None


def _make_client(reconnect: bool = True) -> WebSocketClient:
    """Helper: create a WebSocketClient with a mock WS factory."""
    global _mock_ws_instance

    config = WebSocketClientConfig(
        server_url="ws://localhost:8080",
        token="test-token",
        reconnect=reconnect,
        max_reconnect_delay=300_000,
    )

    def factory(url: str) -> MockWebSocket:
        global _mock_ws_instance
        _mock_ws_instance = MockWebSocket()
        # Emit 'open' in a background thread to simulate async connect
        t = threading.Timer(0.01, lambda: _mock_ws_instance.emit("open"))
        t.daemon = True
        t.start()
        return _mock_ws_instance

    return WebSocketClient(config, ws_factory=factory)


def _wait_for_event(client: WebSocketClient, event: str, timeout: float = 1.0) -> list[Any]:
    """Block until an event fires, returning the args."""
    received: list[Any] = []
    ev = threading.Event()

    def handler(*args: Any) -> None:
        received.extend(args)
        ev.set()

    client.on(event, handler)
    ev.wait(timeout=timeout)
    return received


# ─── Tests ─────────────────────────────────────────────────────────────────────


class TestWebSocketClient:
    def setup_method(self) -> None:
        global _mock_ws_instance
        _mock_ws_instance = None

    def test_emits_connected_on_successful_connection(self) -> None:
        client = _make_client()
        result = _wait_for_event(client, "connected")
        client.connect()
        # connect() fires the open event via timer; wait for connected
        ev = threading.Event()
        client.on("connected", lambda: ev.set())
        # Already connected if result has items (listener registered before connect)
        # Try again with fresh client
        client2 = _make_client()
        ev2 = threading.Event()
        client2.on("connected", lambda: ev2.set())
        client2.connect()
        ev2.wait(timeout=1.0)
        assert client2.is_connected() is True

    def test_sends_serialized_messages(self) -> None:
        client = _make_client()
        ev = threading.Event()
        client.on("connected", lambda: ev.set())
        client.connect()
        ev.wait(timeout=1.0)

        msg = {"type": "disconnect"}
        client.send(msg)

        assert _mock_ws_instance is not None
        assert len(_mock_ws_instance.sent_messages) == 1
        assert json.loads(_mock_ws_instance.sent_messages[0]) == {"type": "disconnect"}

    def test_emits_message_on_received_data(self) -> None:
        client = _make_client()
        connected_ev = threading.Event()
        client.on("connected", lambda: connected_ev.set())
        client.connect()
        connected_ev.wait(timeout=1.0)

        msg_received: list[Any] = []
        msg_ev = threading.Event()

        def on_message(msg: Any) -> None:
            msg_received.append(msg)
            msg_ev.set()

        client.on("message", on_message)

        server_msg = {"type": "authenticated", "entity_id": "ent_123"}
        assert _mock_ws_instance is not None
        _mock_ws_instance.emit("message", json.dumps(server_msg).encode())

        msg_ev.wait(timeout=1.0)
        assert len(msg_received) == 1
        assert msg_received[0] == server_msg

    def test_is_connected_returns_false_before_connect(self) -> None:
        client = _make_client()
        assert client.is_connected() is False

    def test_emits_disconnected_when_connection_closes(self) -> None:
        client = _make_client(reconnect=False)
        connected_ev = threading.Event()
        client.on("connected", lambda: connected_ev.set())
        client.connect()
        connected_ev.wait(timeout=1.0)

        disconnected: list[tuple[int, str]] = []
        disconn_ev = threading.Event()

        def on_disconn(code: int, reason: str) -> None:
            disconnected.append((code, reason))
            disconn_ev.set()

        client.on("disconnected", on_disconn)

        assert _mock_ws_instance is not None
        _mock_ws_instance.emit("close", 1000, b"normal")

        disconn_ev.wait(timeout=1.0)
        assert len(disconnected) == 1
        assert disconnected[0] == (1000, "normal")

    def test_emits_reconnecting_with_exponential_backoff(self) -> None:
        client = _make_client(reconnect=True)
        connected_ev = threading.Event()
        client.on("connected", lambda: connected_ev.set())
        client.connect()
        connected_ev.wait(timeout=1.0)

        reconnect_events: list[dict[str, Any]] = []
        reconnect_ev = threading.Event()

        def on_reconnecting(attempt: int, delay: float) -> None:
            reconnect_events.append({"attempt": attempt, "delay": delay})
            reconnect_ev.set()

        client.on("reconnecting", on_reconnecting)

        # Simulate unexpected disconnect
        assert _mock_ws_instance is not None
        _mock_ws_instance.emit("close", 1006, b"abnormal")

        reconnect_ev.wait(timeout=1.0)

        # Cancel the actual reconnect timer to avoid side effects
        if client._reconnect_timer is not None:
            client._reconnect_timer.cancel()

        assert len(reconnect_events) == 1
        assert reconnect_events[0]["attempt"] == 1
        assert reconnect_events[0]["delay"] == 1000  # First delay: 1s

    def test_throws_when_sending_on_closed_connection(self) -> None:
        client = _make_client()
        with pytest.raises(RuntimeError, match="WebSocket is not connected"):
            client.send({"type": "disconnect"})
