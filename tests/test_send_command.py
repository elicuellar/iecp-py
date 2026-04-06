"""Send command tests -- Phase 8.

Tests the one-shot send flow: connect -> auth -> fetch -> lock -> chunk -> commit -> disconnect.
Mirrors packages/cli/tests/commands/send.test.ts exactly.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

import pytest

from iecp_core.cli.commands.send import execute_send, SendOptions


# ─── Mock WebSocket ────────────────────────────────────────────────────────────


class MockWebSocket:
    """Mock WS that auto-responds to IECP protocol messages."""

    OPEN = 1
    CLOSED = 3

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
        parsed = json.loads(data)

        # Auto-respond based on message type (mirroring the TS mock)
        # Use a small delay (not 0.0) to ensure listener is registered before response fires
        def respond() -> None:
            msg_type = parsed.get("type")
            if msg_type == "authenticate":
                self.emit(
                    "message",
                    json.dumps({"type": "authenticated", "entity_id": "ent_test"}).encode(),
                )
            elif msg_type == "fetch_unread_batch":
                self.emit(
                    "message",
                    json.dumps({
                        "type": "unread_batch",
                        "request_id": parsed.get("request_id"),
                        "payload": {"unread_messages": []},
                    }).encode(),
                )
            elif msg_type == "acquire_speaking_lock":
                self.emit(
                    "message",
                    json.dumps({
                        "type": "lock_acquired",
                        "request_id": parsed.get("request_id"),
                        "granted": True,
                        "ttl_ms": 30000,
                    }).encode(),
                )
            elif msg_type == "append_stream_chunk":
                self.emit(
                    "message",
                    json.dumps({
                        "type": "chunk_ack",
                        "request_id": parsed.get("request_id"),
                        "ok": True,
                    }).encode(),
                )
            elif msg_type == "commit_message":
                self.emit(
                    "message",
                    json.dumps({
                        "type": "commit_response",
                        "request_id": parsed.get("request_id"),
                        "event_id": "evt_committed",
                        "created_at": int(time.time() * 1000),
                    }).encode(),
                )

        t = threading.Timer(0.001, respond)
        t.daemon = True
        t.start()

    def close(self, code: int = 1000, reason: str = "") -> None:
        self.readyState = MockWebSocket.CLOSED
        self.emit("close", code, (reason.encode() if isinstance(reason, str) else reason))

    def terminate(self) -> None:
        self.readyState = MockWebSocket.CLOSED


_mock_ws_instance: MockWebSocket | None = None


class TestSendCommand:
    def setup_method(self) -> None:
        global _mock_ws_instance
        _mock_ws_instance = None

    def test_executes_full_send_flow(self) -> None:
        """connect -> auth -> fetch -> lock -> chunk -> commit -> disconnect"""
        global _mock_ws_instance

        def ws_factory(url: str) -> MockWebSocket:
            global _mock_ws_instance
            ws = MockWebSocket()
            _mock_ws_instance = ws
            # Emit 'open' after brief delay so wiring is done first
            t = threading.Timer(0.001, lambda: ws.emit("open"))
            t.daemon = True
            t.start()
            return ws

        execute_send(
            "Hello IECP!",
            SendOptions(
                server="ws://localhost:8080",
                token="test-token",
                room="conv_test",
            ),
            ws_factory=ws_factory,
        )

        assert _mock_ws_instance is not None
        messages = [json.loads(m)["type"] for m in _mock_ws_instance.sent_messages]

        assert messages == [
            "authenticate",
            "fetch_unread_batch",
            "acquire_speaking_lock",
            "append_stream_chunk",
            "commit_message",
            "disconnect",  # graceful disconnect
        ]

        # Verify the chunk text
        chunk_msg = json.loads(_mock_ws_instance.sent_messages[3])
        assert chunk_msg["text"] == "Hello IECP!"
