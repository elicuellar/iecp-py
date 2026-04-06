"""Connection Manager -- Phase 7 of the IECP protocol.

Manages active WebSocket connections, subscriptions, and client lookups.
"""

from __future__ import annotations

from ..types.entity import EntityId
from ..types.event import ConversationId
from .types import DEFAULT_GATEWAY_CONFIG, GatewayClient, GatewayConfig


class ConnectionManager:
    """Manages active WebSocket connections, subscriptions, and client lookups."""

    def __init__(self, config: GatewayConfig | None = None) -> None:
        self._config: GatewayConfig = config or DEFAULT_GATEWAY_CONFIG
        self._clients: dict[str, GatewayClient] = {}
        # entityId -> clientId
        self._entity_index: dict[str, str] = {}
        # conversationId -> set[clientId]
        self._subscription_index: dict[str, set[str]] = {}

    def add_client(self, client: GatewayClient) -> None:
        """Register a new connected client."""
        self._clients[client.id] = client
        self._entity_index[str(client.entity_id)] = client.id

        # Index existing subscriptions
        for conv_id in client.conversation_ids:
            self._add_to_subscription_index(conv_id, client.id)

    def remove_client(self, client_id: str) -> None:
        """Remove a client by connection ID."""
        client = self._clients.get(client_id)
        if not client:
            return

        # Clean up subscription index
        for conv_id in client.conversation_ids:
            self._remove_from_subscription_index(conv_id, client_id)

        self._entity_index.pop(str(client.entity_id), None)
        del self._clients[client_id]

    def get_client(self, client_id: str) -> GatewayClient | None:
        """Get a client by connection ID."""
        return self._clients.get(client_id)

    def get_client_by_entity(self, entity_id: EntityId) -> GatewayClient | None:
        """Get a client by entity ID."""
        client_id = self._entity_index.get(str(entity_id))
        if client_id is None:
            return None
        return self._clients.get(client_id)

    def get_subscribers(self, conversation_id: ConversationId) -> list[GatewayClient]:
        """Get all clients subscribed to a conversation."""
        client_ids = self._subscription_index.get(str(conversation_id))
        if not client_ids:
            return []

        result: list[GatewayClient] = []
        for client_id in client_ids:
            client = self._clients.get(client_id)
            if client:
                result.append(client)
        return result

    def subscribe(self, client_id: str, conversation_ids: list[ConversationId]) -> None:
        """Subscribe a client to one or more conversations."""
        client = self._clients.get(client_id)
        if not client:
            return

        for conv_id in conversation_ids:
            client.conversation_ids.add(conv_id)
            self._add_to_subscription_index(conv_id, client_id)

    def unsubscribe(self, client_id: str, conversation_ids: list[ConversationId]) -> None:
        """Unsubscribe a client from one or more conversations."""
        client = self._clients.get(client_id)
        if not client:
            return

        for conv_id in conversation_ids:
            client.conversation_ids.discard(conv_id)
            self._remove_from_subscription_index(conv_id, client_id)

    def get_all_clients(self) -> list[GatewayClient]:
        """Get all connected clients."""
        return list(self._clients.values())

    def get_connected_daemons(self) -> list[EntityId]:
        """Get entity IDs of all connected daemons."""
        return [
            client.entity_id
            for client in self._clients.values()
            if client.type == "daemon"
        ]

    def destroy(self) -> None:
        """Clean up all state."""
        self._clients.clear()
        self._entity_index.clear()
        self._subscription_index.clear()

    # -- Private Helpers -----------------------------------------------------

    def _add_to_subscription_index(
        self, conversation_id: ConversationId, client_id: str
    ) -> None:
        key = str(conversation_id)
        if key not in self._subscription_index:
            self._subscription_index[key] = set()
        self._subscription_index[key].add(client_id)

    def _remove_from_subscription_index(
        self, conversation_id: ConversationId, client_id: str
    ) -> None:
        key = str(conversation_id)
        s = self._subscription_index.get(key)
        if s is None:
            return
        s.discard(client_id)
        if len(s) == 0:
            del self._subscription_index[key]
