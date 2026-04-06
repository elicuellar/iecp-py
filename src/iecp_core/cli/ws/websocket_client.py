"""WebSocket Client -- persistent connection to the IECP server.

Features:
- Auto-reconnection with exponential backoff (1s -> 5min cap)
- Ping/pong handling
- Graceful shutdown
- Type-safe message serialization

Mirrors packages/cli/src/ws/WebSocketClient.ts exactly.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class WebSocketClientConfig:
    """Configuration for the WebSocket client."""

    server_url: str
    """WebSocket server URL."""

    token: str
    """Authentication token."""

    reconnect: bool = True
    """Enable auto-reconnection (default: True)."""

    max_reconnect_delay: float = 300_000
    """Maximum reconnection delay in ms (default: 300_000 = 5min)."""


# WebSocket ready-state constants (mirrors ws library)
WS_OPEN = 1
WS_CLOSED = 3


class WebSocketClient:
    """Persistent WebSocket connection to the IECP server with auto-reconnect.

    Uses an EventEmitter-style listener pattern. The WebSocket implementation
    is injectable via the ws_factory parameter for testing.
    """

    def __init__(
        self,
        config: WebSocketClientConfig,
        ws_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._config = config
        # ws_factory(url) -> ws instance  (allows injection for testing)
        self._ws_factory = ws_factory
        self._ws: Any | None = None
        self._reconnect_attempt: int = 0
        self._reconnect_timer: threading.Timer | None = None
        self._intentional_close: bool = False

        # EventEmitter pattern: event_name -> [listeners]
        self._listeners: dict[str, list[Callable[..., Any]]] = {}

    # ─── EventEmitter ──────────────────────────────────────────────────────────

    def on(self, event: str, listener: Callable[..., Any]) -> None:
        """Register a listener for an event."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def off(self, event: str, listener: Callable[..., Any]) -> None:
        """Remove a listener."""
        listeners = self._listeners.get(event)
        if listeners and listener in listeners:
            listeners.remove(listener)

    def remove_listener(self, event: str, listener: Callable[..., Any]) -> None:
        """Alias for off()."""
        self.off(event, listener)

    def emit(self, event: str, *args: Any) -> None:
        """Emit an event to all registered listeners."""
        for listener in list(self._listeners.get(event, [])):
            listener(*args)

    # ─── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect to the IECP server.

        Creates the WebSocket and wires event handlers.
        The actual connect is synchronous — resolution happens when the
        'open' event is emitted by the underlying WebSocket.
        """
        self._intentional_close = False
        self._do_connect()

    def _do_connect(self) -> None:
        """Create the WebSocket and wire events."""
        if self._ws_factory is None:
            raise RuntimeError("ws_factory not configured — inject one for testing")
        ws = self._ws_factory(self._config.server_url)
        self._ws = ws

        # Wire event handlers onto the ws mock/real object
        ws.on("open", self._handle_open)
        ws.on("message", self._handle_message)
        ws.on("close", self._handle_close)
        ws.on("error", self._handle_error)

    def _handle_open(self) -> None:
        self._reconnect_attempt = 0
        self.emit("connected")

    def _handle_message(self, data: Any) -> None:
        try:
            raw = data if isinstance(data, str) else data.decode("utf-8")
            message = json.loads(raw)
            self.emit("message", message)
        except Exception as err:
            self.emit("error", Exception(f"Failed to parse server message: {err}"))

    def _handle_close(self, code: int, reason: Any) -> None:
        self._ws = None
        reason_str = (
            reason.decode("utf-8")
            if isinstance(reason, (bytes, bytearray))
            else str(reason)
        )
        self.emit("disconnected", code, reason_str)

        if not self._intentional_close and self._config.reconnect:
            self._schedule_reconnect()

    def _handle_error(self, err: Exception) -> None:
        self.emit("error", err)

    def disconnect(self) -> None:
        """Gracefully disconnect from the server."""
        self._intentional_close = True

        if self._reconnect_timer is not None:
            self._reconnect_timer.cancel()
            self._reconnect_timer = None

        if self._ws is not None and self.is_connected():
            self.send({"type": "disconnect"})
            self._ws.close(1000, "client disconnect")

    def send(self, message: Any) -> None:
        """Send a message to the server.

        Accepts a dict or a pydantic model.
        """
        if not self.is_connected():
            raise RuntimeError("WebSocket is not connected")

        if hasattr(message, "model_dump"):
            payload = message.model_dump(exclude_none=True)
        elif isinstance(message, dict):
            payload = message
        else:
            raise TypeError(f"Cannot serialize message of type {type(message)}")

        self._ws.send(json.dumps(payload))  # type: ignore[union-attr]

    def is_connected(self) -> bool:
        """Check if the client is currently connected."""
        if self._ws is None:
            return False
        return getattr(self._ws, "readyState", WS_CLOSED) == WS_OPEN

    # ─── Reconnection ──────────────────────────────────────────────────────────

    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection with exponential backoff."""
        self._reconnect_attempt += 1
        base_delay = 1000.0  # 1 second in ms
        delay_ms = min(
            base_delay * (2 ** (self._reconnect_attempt - 1)),
            self._config.max_reconnect_delay,
        )

        self.emit("reconnecting", self._reconnect_attempt, delay_ms)

        def _reconnect() -> None:
            self._reconnect_timer = None
            try:
                self._do_connect()
            except Exception as err:
                self.emit("error", err if isinstance(err, Exception) else Exception(str(err)))

        self._reconnect_timer = threading.Timer(delay_ms / 1000.0, _reconnect)
        self._reconnect_timer.daemon = True
        self._reconnect_timer.start()
