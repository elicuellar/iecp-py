"""WebSocket Gateway -- Phase 7 of the IECP protocol.

The main gateway class that delivers real-time events to humans
and daemon CLIs via WebSocket (§3.2, §10.1).

Event flow:
1. Human sends message -> API -> Event -> Orchestrator -> Gateway broadcasts
2. Debounce seals batch -> Gateway sends batch_sealed to daemons
3. Orchestrator dispatches -> Gateway sends dispatch to target daemon
4. Daemon streams chunks -> Gateway broadcasts to human clients
5. Daemon commits -> Gateway broadcasts the committed event
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..debounce.types import SealedBatch
from ..lock.types import LockState
from ..orchestrator.types import DispatchPayload
from ..types.entity import EntityId
from ..types.event import ConversationId, Event
from ..utils import generate_id
from .connection_manager import ConnectionManager
from .daemon_buffer import DaemonBuffer
from .types import (
    DEFAULT_GATEWAY_CONFIG,
    ActiveSignal,
    GatewayClient,
    GatewayConfig,
    TokenValidator,
)

# WebSocket readyState constants
WS_OPEN = 1
WS_CLOSED = 3


class WebSocketGateway:
    """Delivers real-time events to humans and daemon CLIs via WebSocket."""

    def __init__(
        self,
        connection_manager: ConnectionManager,
        daemon_buffer: DaemonBuffer,
        token_validator: TokenValidator,
        config: GatewayConfig | None = None,
    ) -> None:
        self._connection_manager = connection_manager
        self._daemon_buffer = daemon_buffer
        self._token_validator = token_validator
        self._config: GatewayConfig = config or DEFAULT_GATEWAY_CONFIG

        # Event listeners (EventEmitter pattern)
        self._listeners: dict[str, list[Callable[..., Any]]] = {}

    # -- EventEmitter pattern ------------------------------------------------

    def on(self, event: str, listener: Callable[..., Any]) -> None:
        """Register a listener for gateway events."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def off(self, event: str, listener: Callable[..., Any]) -> None:
        """Remove a listener."""
        listeners = self._listeners.get(event)
        if listeners and listener in listeners:
            listeners.remove(listener)

    def _emit(self, event: str, *args: Any) -> None:
        """Emit an event to all registered listeners."""
        for listener in self._listeners.get(event, []):
            listener(*args)

    # -- Broadcast / Send ----------------------------------------------------

    def broadcast(
        self,
        conversation_id: ConversationId,
        message: dict[str, Any],
        exclude_client_id: str | None = None,
    ) -> None:
        """Broadcast a message to all subscribers of a conversation."""
        subscribers = self._connection_manager.get_subscribers(conversation_id)
        data = json.dumps(message)

        for client in subscribers:
            if exclude_client_id and client.id == exclude_client_id:
                continue
            if self._ws_is_open(client.ws):
                client.ws.send(data)

    def send_to_entity(self, entity_id: EntityId, message: dict[str, Any]) -> bool:
        """Send a message to a specific entity. Returns False if entity not connected."""
        client = self._connection_manager.get_client_by_entity(entity_id)
        if client is None or not self._ws_is_open(client.ws):
            return False
        client.ws.send(json.dumps(message))
        return True

    # -- Event Handlers (wire to Orchestrator) --------------------------------

    def handle_event(self, event: Event) -> None:
        """Handle a new event -- broadcast to all subscribers of the conversation."""
        self.broadcast(
            event.conversation_id,
            {"type": "event", "payload": event.model_dump()},
        )

    def handle_batch_sealed(self, batch: SealedBatch) -> None:
        """Handle a sealed batch -- notify subscribers."""
        self.broadcast(
            batch.conversation_id,
            {"type": "batch_sealed", "payload": batch.model_dump()},
        )

    def handle_dispatch(self, dispatch: DispatchPayload) -> None:
        """Handle a dispatch -- send to the target daemon/artificer.

        If daemon is disconnected, buffer the event.
        """
        sent = self.send_to_entity(
            dispatch.entity_id,
            {"type": "dispatch", "payload": dispatch.model_dump()},
        )

        if not sent:
            # Buffer for disconnected daemon -- create a synthetic event
            from datetime import datetime, timezone

            synthetic = Event(
                id=dispatch.trace_id,  # type: ignore[arg-type]
                conversation_id=dispatch.conversation_id,
                author_id="system",  # type: ignore[arg-type]
                author_type="system",
                type="system",
                content={
                    "system_event": "dispatch",
                    "description": f"Dispatch to {dispatch.entity_id}",
                    "data": dispatch.model_dump(),
                },
                is_continuation=False,
                is_complete=True,
                ai_depth_counter=dispatch.ai_depth_counter,
                status="active",
                metadata={},
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._daemon_buffer.buffer(dispatch.entity_id, synthetic)

    def handle_lock_state(self, state: LockState) -> None:
        """Handle lock state changes -- broadcast to conversation."""
        self.broadcast(
            state.conversation_id,
            {"type": "lock_state", "payload": state.model_dump()},
        )

    def handle_signal(self, signal: ActiveSignal) -> None:
        """Handle attention signals -- broadcast to conversation."""
        self.broadcast(
            signal.conversation_id,
            {"type": "signal", "payload": signal.model_dump()},
        )

    def handle_stream_chunk(
        self,
        conversation_id: ConversationId,
        entity_id: EntityId,
        text: str,
        chunk_index: int,
    ) -> None:
        """Handle stream chunks -- broadcast to human clients only.

        Stream chunks are ephemeral -- not stored in the event log.
        """
        subscribers = self._connection_manager.get_subscribers(conversation_id)
        message = {
            "type": "stream_chunk",
            "payload": {
                "conversationId": conversation_id,
                "entityId": entity_id,
                "text": text,
                "chunkIndex": chunk_index,
            },
        }
        data = json.dumps(message)

        for client in subscribers:
            # Only send stream chunks to human clients
            if client.type == "human" and self._ws_is_open(client.ws):
                client.ws.send(data)

    # -- Client Message Handling ---------------------------------------------

    def handle_client_message(self, client: GatewayClient, raw: str) -> None:
        """Handle an incoming client message."""
        client.last_ping_at = _now_ms()

        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return  # Ignore unparseable messages

        msg_type = msg.get("type")

        if msg_type == "ping":
            client.ws.send(json.dumps({"type": "pong"}))

        elif msg_type == "subscribe":
            conv_ids = [ConversationId(c) for c in msg.get("conversationIds", [])]
            self._connection_manager.subscribe(client.id, conv_ids)

        elif msg_type == "unsubscribe":
            conv_ids = [ConversationId(c) for c in msg.get("conversationIds", [])]
            self._connection_manager.unsubscribe(client.id, conv_ids)

        elif msg_type == "typing_start":
            conv_id = ConversationId(msg["conversationId"])
            self._emit("typing_start", conv_id, client.entity_id)
            self.broadcast(
                conv_id,
                {
                    "type": "typing",
                    "payload": {
                        "conversationId": conv_id,
                        "entityId": client.entity_id,
                    },
                },
                exclude_client_id=client.id,
            )

        elif msg_type == "stream_chunk":
            self.handle_stream_chunk(
                ConversationId(msg["conversationId"]),
                client.entity_id,
                msg.get("text", ""),
                0,
            )

    def handle_disconnect(self, client: GatewayClient) -> None:
        """Handle client disconnect."""
        self._connection_manager.remove_client(client.id)
        self._emit("client_disconnected", client)

    # -- Heartbeat -----------------------------------------------------------

    def check_heartbeats(self) -> None:
        """Check heartbeat timeouts and ping live clients."""
        import time

        now = time.time() * 1000.0
        timeout = self._config.heartbeat_interval_ms + self._config.heartbeat_timeout_ms

        for client in self._connection_manager.get_all_clients():
            if now - client.last_ping_at > timeout:
                # Client missed heartbeat -- disconnect
                client.ws.close(1001, "Heartbeat timeout")
                self.handle_disconnect(client)
            elif self._ws_is_open(client.ws):
                client.ws.ping()

    # -- Lifecycle -----------------------------------------------------------

    def destroy(self) -> None:
        """Destroy the gateway, cleaning up all state."""
        self._connection_manager.destroy()
        self._daemon_buffer.destroy()

    # -- Private Helpers -----------------------------------------------------

    @staticmethod
    def _ws_is_open(ws: Any) -> bool:
        """Check if a WebSocket is in OPEN state."""
        return getattr(ws, "readyState", WS_OPEN) == WS_OPEN


def _now_ms() -> float:
    import time

    return time.time() * 1000.0
