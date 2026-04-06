"""Daemon Event Buffer -- Phase 7 of the IECP protocol.

Buffers events for disconnected daemons within a TTL window (§14.3).
On reconnect, buffered events are flushed in order.
If the buffer window expires, the daemon's status transitions to LEFT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..types.entity import EntityId
from ..types.event import Event
from .types import DEFAULT_GATEWAY_CONFIG


@dataclass
class _BufferEntry:
    events: list[Event] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time() * 1000.0)


class DaemonBuffer:
    """Buffers events for disconnected daemons with TTL-based expiry."""

    def __init__(
        self,
        ttl_ms: float | None = None,
        max_events: int | None = None,
    ) -> None:
        self._ttl_ms: float = (
            ttl_ms if ttl_ms is not None else DEFAULT_GATEWAY_CONFIG.daemon_buffer_ttl_ms
        )
        self._max_events: int = (
            max_events
            if max_events is not None
            else DEFAULT_GATEWAY_CONFIG.daemon_buffer_max_events
        )
        self._buffers: dict[str, _BufferEntry] = {}

    def buffer(self, entity_id: EntityId, event: Event) -> None:
        """Buffer an event for a disconnected daemon."""
        key = str(entity_id)
        entry = self._buffers.get(key)
        if entry is None:
            entry = _BufferEntry()
            self._buffers[key] = entry

        entry.events.append(event)

        # Enforce max_events by dropping oldest
        while len(entry.events) > self._max_events:
            entry.events.pop(0)

    def flush(self, entity_id: EntityId) -> list[Event]:
        """Flush all buffered events for a daemon (on reconnect).

        Returns events in order and clears the buffer.
        """
        key = str(entity_id)
        entry = self._buffers.pop(key, None)
        if entry is None:
            return []
        return entry.events

    def has_events(self, entity_id: EntityId) -> bool:
        """Check if a daemon has buffered events."""
        entry = self._buffers.get(str(entity_id))
        return entry is not None and len(entry.events) > 0

    def get_buffer_size(self, entity_id: EntityId) -> int:
        """Get the number of buffered events for a daemon."""
        entry = self._buffers.get(str(entity_id))
        return len(entry.events) if entry else 0

    def clear_expired(self) -> list[EntityId]:
        """Remove buffers that have exceeded the TTL.

        Returns entity IDs of expired buffers.
        """
        now = time.time() * 1000.0
        expired: list[EntityId] = []
        for key, entry in list(self._buffers.items()):
            if now - entry.created_at >= self._ttl_ms:
                del self._buffers[key]
                expired.append(EntityId(key))
        return expired

    def clear_buffer(self, entity_id: EntityId) -> None:
        """Clear the buffer for a specific daemon."""
        self._buffers.pop(str(entity_id), None)

    def destroy(self) -> None:
        """Clean up state."""
        self._buffers.clear()
