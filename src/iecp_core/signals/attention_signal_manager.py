"""AttentionSignalManager -- SS18 of the specification.

Manages lightweight attention signals that communicate engagement
without requiring floor lock, triggering dispatch, or resetting cascade.

Rules (SS18.3):
- Max 1 signal per entity per batch (rate limiting)
- Signals have TTL (5 min default), auto-expire
- Newer signal from same entity replaces older in same conversation
- Signals do NOT require floor lock
- Signals do NOT reset cascade counter
- Signals do NOT trigger batch dispatch
"""

from __future__ import annotations

import time
from typing import Optional

from ..types.entity import EntityId
from ..types.event import BatchId, ConversationId
from .types import (
    ActiveSignal,
    AttentionSignalConfig,
    AttentionSignalType,
    DEFAULT_ATTENTION_SIGNAL_CONFIG,
)


def _rate_limit_key(entity_id: EntityId, batch_id: BatchId) -> str:
    return f"{entity_id}::{batch_id}"


def _signal_key(conversation_id: ConversationId, entity_id: EntityId) -> str:
    return f"{conversation_id}::{entity_id}"


class AttentionSignalManager:
    """Manages attention signals for entities in conversations."""

    def __init__(self, config: Optional[AttentionSignalConfig] = None) -> None:
        self._config = config or DEFAULT_ATTENTION_SIGNAL_CONFIG
        self._signals: dict[str, ActiveSignal] = {}
        self._batch_signal_counts: dict[str, int] = {}
        self._destroyed = False

    def signal(
        self,
        *,
        entity_id: EntityId,
        conversation_id: ConversationId,
        signal_type: AttentionSignalType,
        batch_id: Optional[BatchId] = None,
        note: Optional[str] = None,
    ) -> bool:
        """Register a signal. Returns False if rate-limited or destroyed."""
        if self._destroyed:
            return False

        # Rate limit: max signals per entity per batch
        if batch_id is not None:
            rl_key = _rate_limit_key(entity_id, batch_id)
            count = self._batch_signal_counts.get(rl_key, 0)
            if count >= self._config.max_signals_per_batch:
                return False
            self._batch_signal_counts[rl_key] = count + 1

        now = time.time() * 1000  # ms
        active_signal = ActiveSignal(
            entity_id=entity_id,
            conversation_id=conversation_id,
            signal_type=signal_type,
            note=note,
            batch_id=batch_id,
            created_at=now,
            expires_at=now + self._config.ttl_ms,
        )

        # Newer signal replaces older for same entity+conversation
        key = _signal_key(conversation_id, entity_id)
        self._signals[key] = active_signal
        return True

    def get_signals(self, conversation_id: ConversationId) -> list[ActiveSignal]:
        """Get all active (non-expired) signals for a conversation."""
        now = time.time() * 1000
        return [
            sig
            for sig in self._signals.values()
            if sig.conversation_id == conversation_id and sig.expires_at > now
        ]

    def get_entity_signal(
        self, conversation_id: ConversationId, entity_id: EntityId
    ) -> Optional[ActiveSignal]:
        """Get signal for a specific entity in a conversation."""
        key = _signal_key(conversation_id, entity_id)
        sig = self._signals.get(key)
        if sig is None:
            return None
        if sig.expires_at <= time.time() * 1000:
            del self._signals[key]
            return None
        return sig

    def clear_signal(
        self, conversation_id: ConversationId, entity_id: EntityId
    ) -> None:
        """Clear signal for a specific entity in a conversation."""
        key = _signal_key(conversation_id, entity_id)
        self._signals.pop(key, None)

    def clear_expired(self) -> None:
        """Clear all expired signals."""
        now = time.time() * 1000
        expired_keys = [
            key for key, sig in self._signals.items() if sig.expires_at <= now
        ]
        for key in expired_keys:
            del self._signals[key]

    def destroy(self) -> None:
        """Destroy the manager -- clear all state."""
        self._destroyed = True
        self._signals.clear()
        self._batch_signal_counts.clear()
