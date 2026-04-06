"""Metrics Collector -- §17.1 of the IECP specification.

Event-driven collector that attaches to Orchestrator, ArtificerRuntime,
and WebSocketGateway event emitters. Collects operational metrics
for per-conversation, per-entity, and system-wide reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..types.entity import EntityId
from ..types.event import ConversationId
from .types import (
    ConversationMetrics,
    EntityMetrics,
    PercentileStats,
    SystemMetrics,
)


# --- Internal Accumulators --------------------------------------------------


@dataclass
class _ConversationAccumulator:
    event_count: int = 0
    events_by_type: dict[str, int] = field(default_factory=dict)
    events_by_entity: dict[str, int] = field(default_factory=dict)
    lock_acquisitions: int = 0
    lock_denials: int = 0
    lock_hold_durations: list[float] = field(default_factory=list)
    max_cascade_depth: int = 0
    batch_message_counts: list[int] = field(default_factory=list)
    decisions_proposed: int = 0
    decisions_affirmed: int = 0
    handoff_count: int = 0


@dataclass
class _EntityAccumulator:
    response_latencies: list[float] = field(default_factory=list)
    yield_count: int = 0
    dispatch_count: int = 0
    lock_timeouts: int = 0
    lock_attempts: int = 0
    disconnection_count: int = 0
    total_disconnected_ms: float = 0.0
    last_disconnect_at: float | None = None
    signals_by_type: dict[str, int] = field(default_factory=dict)


# --- Percentile Helper ------------------------------------------------------


def _compute_percentiles(values: list[float]) -> PercentileStats:
    if not values:
        return PercentileStats(p50=0.0, p95=0.0, p99=0.0)
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(pct: float) -> float:
        idx = max(0, int((pct / 100.0) * n + 0.5) - 1)
        return sorted_vals[idx]

    return PercentileStats(p50=percentile(50), p95=percentile(95), p99=percentile(99))


# --- MetricsCollector -------------------------------------------------------


class MetricsCollector:
    """Collects IECP operational metrics by subscribing to emitter events."""

    def __init__(self) -> None:
        self._conversations: dict[str, _ConversationAccumulator] = {}
        self._entities: dict[str, _EntityAccumulator] = {}
        self._start_time: float = time.time() * 1000
        self._active_connections: int = 0
        self._artificer_queue_depth: int = 0
        self._dispatch_timestamps: dict[str, float] = {}

    def attach(
        self,
        *,
        orchestrator: Any,
        runtime: Any,
        gateway: Any,
    ) -> None:
        """Subscribe to orchestrator, runtime, and gateway events."""

        # Orchestrator events
        orchestrator.on("trace", self._handle_trace)
        orchestrator.on("cascade_limit", self._handle_cascade_limit)

        # Runtime events
        runtime.on("message_committed", self._handle_message_committed)

        # Gateway events
        gateway.on("client_connected", self._handle_client_connected)
        gateway.on("client_disconnected", self._handle_client_disconnected)

    def get_conversation_metrics(self, conversation_id: ConversationId) -> ConversationMetrics:
        acc = self._conversations.get(str(conversation_id))
        if acc is None:
            return ConversationMetrics()

        total_lock_attempts = acc.lock_acquisitions + acc.lock_denials
        avg_hold = (
            sum(acc.lock_hold_durations) / len(acc.lock_hold_durations)
            if acc.lock_hold_durations
            else 0.0
        )
        multi_batches = sum(1 for c in acc.batch_message_counts if c >= 2)
        debounce_eff = (
            multi_batches / len(acc.batch_message_counts)
            if acc.batch_message_counts
            else 0.0
        )

        return ConversationMetrics(
            event_count=acc.event_count,
            events_by_type=dict(acc.events_by_type),
            events_by_entity=dict(acc.events_by_entity),
            lock_acquisitions=acc.lock_acquisitions,
            lock_denials=acc.lock_denials,
            lock_grant_ratio=(
                acc.lock_acquisitions / total_lock_attempts if total_lock_attempts > 0 else 0.0
            ),
            avg_lock_hold_duration_ms=avg_hold,
            max_cascade_depth=acc.max_cascade_depth,
            debounce_efficiency=debounce_eff,
            avg_context_payload_tokens=0.0,
            decisions_proposed=acc.decisions_proposed,
            decisions_affirmed=acc.decisions_affirmed,
            handoff_count=acc.handoff_count,
        )

    def get_entity_metrics(self, entity_id: EntityId) -> EntityMetrics:
        acc = self._entities.get(str(entity_id))
        if acc is None:
            return EntityMetrics()

        total_dispatch = acc.dispatch_count + acc.yield_count
        yield_rate = acc.yield_count / total_dispatch if total_dispatch > 0 else 0.0
        lock_timeout_rate = (
            acc.lock_timeouts / acc.lock_attempts if acc.lock_attempts > 0 else 0.0
        )

        return EntityMetrics(
            response_latency_ms=_compute_percentiles(acc.response_latencies),
            yield_rate=yield_rate,
            lock_timeout_rate=lock_timeout_rate,
            disconnection_count=acc.disconnection_count,
            total_disconnected_ms=acc.total_disconnected_ms,
            signals_by_type=dict(acc.signals_by_type),
        )

    def get_system_metrics(self) -> SystemMetrics:
        total_events = sum(acc.event_count for acc in self._conversations.values())
        return SystemMetrics(
            total_conversations=len(self._conversations),
            total_events=total_events,
            active_connections=self._active_connections,
            artificer_queue_depth=self._artificer_queue_depth,
            uptime_ms=time.time() * 1000 - self._start_time,
        )

    def set_queue_depth(self, depth: int) -> None:
        self._artificer_queue_depth = depth

    def record_decision_proposed(self, conversation_id: ConversationId) -> None:
        self._get_conv(str(conversation_id)).decisions_proposed += 1

    def record_decision_affirmed(self, conversation_id: ConversationId) -> None:
        self._get_conv(str(conversation_id)).decisions_affirmed += 1

    def record_handoff(self, conversation_id: ConversationId) -> None:
        self._get_conv(str(conversation_id)).handoff_count += 1

    def record_signal(self, entity_id: EntityId, signal_type: str) -> None:
        entity = self._get_entity(str(entity_id))
        entity.signals_by_type[signal_type] = entity.signals_by_type.get(signal_type, 0) + 1

    def reset(self) -> None:
        self._conversations.clear()
        self._entities.clear()
        self._dispatch_timestamps.clear()
        self._active_connections = 0
        self._artificer_queue_depth = 0

    # --- Private ------------------------------------------------------------

    def _handle_trace(self, trace: Any) -> None:
        conv_id = str(trace.conversation_id)
        conv = self._get_conv(conv_id)

        # Lock tracking
        if trace.lock_result is not None:
            if trace.lock_result.granted:
                conv.lock_acquisitions += 1
            else:
                conv.lock_denials += 1

        # Dispatch tracking
        if trace.outcome == "dispatched" and trace.dispatch_entity is not None:
            key = f"{conv_id}::{trace.dispatch_entity}"
            self._dispatch_timestamps[key] = time.time() * 1000
            entity = self._get_entity(str(trace.dispatch_entity))
            entity.dispatch_count += 1

        # Yield tracking
        if trace.outcome == "gated" and trace.routing.selected_entity is not None:
            entity = self._get_entity(str(trace.routing.selected_entity))
            entity.yield_count += 1

    def _handle_cascade_limit(self, conversation_id: Any, depth: int) -> None:
        conv = self._get_conv(str(conversation_id))
        conv.max_cascade_depth = max(conv.max_cascade_depth, depth)

    def _handle_message_committed(self, evt: Any) -> None:
        # evt may be a dict or object with conversation_id and entity_id
        if isinstance(evt, dict):
            conv_id = str(evt.get("conversation_id", evt.get("conversationId", "")))
            entity_id = str(evt.get("entity_id", evt.get("entityId", "")))
        else:
            conv_id = str(getattr(evt, "conversation_id", getattr(evt, "conversationId", "")))
            entity_id = str(getattr(evt, "entity_id", getattr(evt, "entityId", "")))

        key = f"{conv_id}::{entity_id}"
        dispatch_time = self._dispatch_timestamps.pop(key, None)
        if dispatch_time is not None:
            latency = time.time() * 1000 - dispatch_time
            entity = self._get_entity(entity_id)
            entity.response_latencies.append(latency)

        conv = self._get_conv(conv_id)
        conv.event_count += 1
        conv.events_by_type["message"] = conv.events_by_type.get("message", 0) + 1
        conv.events_by_entity[entity_id] = conv.events_by_entity.get(entity_id, 0) + 1

    def _handle_client_connected(self, client: Any = None) -> None:
        self._active_connections += 1

    def _handle_client_disconnected(self, client: Any = None) -> None:
        self._active_connections = max(0, self._active_connections - 1)
        if client is not None:
            if isinstance(client, dict):
                client_type = client.get("type", "")
                entity_id = str(client.get("entity_id", client.get("entityId", "")))
            else:
                client_type = getattr(client, "type", "")
                entity_id = str(getattr(client, "entity_id", getattr(client, "entityId", "")))

            if client_type == "daemon" and entity_id:
                entity = self._get_entity(entity_id)
                entity.disconnection_count += 1
                entity.last_disconnect_at = time.time() * 1000

    def _get_conv(self, conv_id: str) -> _ConversationAccumulator:
        if conv_id not in self._conversations:
            self._conversations[conv_id] = _ConversationAccumulator()
        return self._conversations[conv_id]

    def _get_entity(self, entity_id: str) -> _EntityAccumulator:
        if entity_id not in self._entities:
            self._entities[entity_id] = _EntityAccumulator()
        return self._entities[entity_id]
