"""Attention Signal Types -- SS18 of the specification.

Lightweight, non-conversational indicators that communicate
engagement without generating noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ..types.entity import EntityId
from ..types.event import BatchId, ConversationId

# -- Signal Types --------------------------------------------------------------

AttentionSignalType = Literal["listening", "thinking", "deferred", "acknowledged"]

# -- Configuration -------------------------------------------------------------


@dataclass(frozen=True)
class AttentionSignalConfig:
    """Configuration for the AttentionSignalManager."""

    ttl_ms: int = 300_000
    """TTL for signals in milliseconds (default: 5 minutes)."""

    max_signals_per_batch: int = 1
    """Max signals per entity per batch (default: 1)."""


DEFAULT_ATTENTION_SIGNAL_CONFIG = AttentionSignalConfig()


# -- Active Signal -------------------------------------------------------------


@dataclass
class ActiveSignal:
    """An active attention signal."""

    entity_id: EntityId
    conversation_id: ConversationId
    signal_type: AttentionSignalType
    created_at: float
    expires_at: float
    note: Optional[str] = None
    batch_id: Optional[BatchId] = None
