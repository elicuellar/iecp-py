"""Observability Types -- Phase 10 of the IECP protocol.

Metric shapes for conversations, entities, and system-wide monitoring (§17).
"""

from __future__ import annotations

from pydantic import BaseModel


# --- Conversation Metrics ---------------------------------------------------


class ConversationMetrics(BaseModel):
    event_count: int = 0
    events_by_type: dict[str, int] = {}
    events_by_entity: dict[str, int] = {}
    lock_acquisitions: int = 0
    lock_denials: int = 0
    lock_grant_ratio: float = 0.0
    avg_lock_hold_duration_ms: float = 0.0
    max_cascade_depth: int = 0
    debounce_efficiency: float = 0.0
    avg_context_payload_tokens: float = 0.0
    decisions_proposed: int = 0
    decisions_affirmed: int = 0
    handoff_count: int = 0


# --- Entity Metrics ---------------------------------------------------------


class PercentileStats(BaseModel):
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class EntityMetrics(BaseModel):
    response_latency_ms: PercentileStats = PercentileStats()
    yield_rate: float = 0.0
    lock_timeout_rate: float = 0.0
    disconnection_count: int = 0
    total_disconnected_ms: float = 0.0
    signals_by_type: dict[str, int] = {}


# --- System Metrics ---------------------------------------------------------


class SystemMetrics(BaseModel):
    total_conversations: int = 0
    total_events: int = 0
    active_connections: int = 0
    artificer_queue_depth: int = 0
    uptime_ms: float = 0.0
