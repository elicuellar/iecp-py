"""MetricsCollector tests -- Phase 10: Observability."""

from __future__ import annotations

import time
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

from iecp_core.observability import MetricsCollector
from iecp_core.orchestrator.types import GatingResult, OrchestrationTrace, RoutingDecision
from iecp_core.lock.types import LockResult, LockState
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId, EventId


# --- Helpers ----------------------------------------------------------------


class FakeEmitter:
    """Simple event emitter for testing."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable]] = {}

    def on(self, event: str, listener: Callable) -> None:
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def emit(self, event: str, *args: Any) -> None:
        for listener in self._listeners.get(event, []):
            listener(*args)


def make_trace(**overrides: Any) -> OrchestrationTrace:
    routing = overrides.pop(
        "routing",
        RoutingDecision(
            eligible_entities=[],
            selected_entity=EntityId("entity-1"),
            reason="test",
            rule_applied="explicit_mention",
        ),
    )
    gating = overrides.pop("gating", GatingResult(allowed=True, checks=[]))

    defaults: dict[str, Any] = {
        "trace_id": f"trace-{id(object())}",
        "conversation_id": ConversationId("conv-1"),
        "timestamp": time.time() * 1000,
        "outcome": "dispatched",
        "duration_ms": 10.0,
        "dispatch_entity": EntityId("entity-1"),
    }
    defaults.update(overrides)

    return OrchestrationTrace(
        routing=routing,
        gating=gating,
        **defaults,
    )


# --- Tests ------------------------------------------------------------------


class TestMetricsCollector:
    def setup_method(self) -> None:
        self.collector = MetricsCollector()
        self.orchestrator = FakeEmitter()
        self.runtime = FakeEmitter()
        self.gateway = FakeEmitter()
        self.collector.attach(
            orchestrator=self.orchestrator,
            runtime=self.runtime,
            gateway=self.gateway,
        )

    def test_collects_lock_acquisitions(self) -> None:
        trace = make_trace(
            conversation_id=ConversationId("conv-1"),
            lock_result=LockResult(
                granted=True,
                lock=LockState(
                    holder_id=EntityId("entity-1"),
                    conversation_id=ConversationId("conv-1"),
                    acquired_at=1000.0,
                    ttl_ms=30000.0,
                    estimated_ms=30000.0,
                    expires_at=31000.0,
                ),
            ),
        )
        self.orchestrator.emit("trace", trace)

        metrics = self.collector.get_conversation_metrics(ConversationId("conv-1"))
        assert metrics.lock_acquisitions == 1
        assert metrics.lock_denials == 0

    def test_tracks_lock_denials(self) -> None:
        trace = make_trace(
            lock_result=LockResult(granted=False, reason="held by other"),
            outcome="gated",
        )
        self.orchestrator.emit("trace", trace)

        metrics = self.collector.get_conversation_metrics(ConversationId("conv-1"))
        assert metrics.lock_denials == 1
        assert metrics.lock_grant_ratio == 0.0

    def test_calculates_lock_grant_ratio(self) -> None:
        granted = LockResult(
            granted=True,
            lock=LockState(
                holder_id=EntityId("e"),
                conversation_id=ConversationId("conv-1"),
                acquired_at=0,
                ttl_ms=30000,
                estimated_ms=30000,
                expires_at=30000,
            ),
        )
        denied = LockResult(granted=False, reason="held")

        self.orchestrator.emit("trace", make_trace(lock_result=granted))
        self.orchestrator.emit("trace", make_trace(lock_result=granted))
        self.orchestrator.emit("trace", make_trace(lock_result=denied, outcome="gated"))

        metrics = self.collector.get_conversation_metrics(ConversationId("conv-1"))
        assert metrics.lock_acquisitions == 2
        assert metrics.lock_denials == 1
        assert abs(metrics.lock_grant_ratio - (2 / 3)) < 0.01

    def test_tracks_entity_response_latency(self) -> None:
        trace = make_trace(
            dispatch_entity=EntityId("e-1"),
            outcome="dispatched",
        )
        self.orchestrator.emit("trace", trace)

        # Simulate a small delay
        time.sleep(0.01)

        self.runtime.emit(
            "message_committed",
            {"conversation_id": "conv-1", "entity_id": "e-1"},
        )

        metrics = self.collector.get_entity_metrics(EntityId("e-1"))
        assert metrics.response_latency_ms.p50 >= 5

    def test_tracks_cascade_depth(self) -> None:
        self.orchestrator.emit("cascade_limit", ConversationId("conv-1"), 3)
        metrics = self.collector.get_conversation_metrics(ConversationId("conv-1"))
        assert metrics.max_cascade_depth == 3

    def test_tracks_cascade_max_depth(self) -> None:
        self.orchestrator.emit("cascade_limit", ConversationId("conv-1"), 2)
        self.orchestrator.emit("cascade_limit", ConversationId("conv-1"), 5)
        self.orchestrator.emit("cascade_limit", ConversationId("conv-1"), 3)
        metrics = self.collector.get_conversation_metrics(ConversationId("conv-1"))
        assert metrics.max_cascade_depth == 5

    def test_tracks_decisions_and_handoffs(self) -> None:
        conv = ConversationId("conv-1")
        self.collector.record_decision_proposed(conv)
        self.collector.record_decision_proposed(conv)
        self.collector.record_decision_affirmed(conv)
        self.collector.record_handoff(conv)

        metrics = self.collector.get_conversation_metrics(conv)
        assert metrics.decisions_proposed == 2
        assert metrics.decisions_affirmed == 1
        assert metrics.handoff_count == 1

    def test_tracks_gateway_connections(self) -> None:
        self.gateway.emit("client_connected", {"entity_id": "e-1", "type": "human"})
        self.gateway.emit("client_connected", {"entity_id": "e-2", "type": "daemon"})

        sys = self.collector.get_system_metrics()
        assert sys.active_connections == 2

        self.gateway.emit("client_disconnected", {"entity_id": "e-2", "type": "daemon"})
        sys = self.collector.get_system_metrics()
        assert sys.active_connections == 1

    def test_tracks_daemon_disconnections(self) -> None:
        self.gateway.emit("client_disconnected", {"entity_id": "e-1", "type": "daemon"})
        metrics = self.collector.get_entity_metrics(EntityId("e-1"))
        assert metrics.disconnection_count == 1

    def test_non_daemon_disconnections_not_tracked_on_entity(self) -> None:
        self.gateway.emit("client_connected", {"entity_id": "e-1", "type": "human"})
        self.gateway.emit("client_disconnected", {"entity_id": "e-1", "type": "human"})
        metrics = self.collector.get_entity_metrics(EntityId("e-1"))
        # active connections decreased but entity disconnection_count not incremented
        assert metrics.disconnection_count == 0

    def test_tracks_signal_usage(self) -> None:
        self.collector.record_signal(EntityId("e-1"), "thinking")
        self.collector.record_signal(EntityId("e-1"), "thinking")
        self.collector.record_signal(EntityId("e-1"), "listening")

        metrics = self.collector.get_entity_metrics(EntityId("e-1"))
        assert metrics.signals_by_type.get("thinking") == 2
        assert metrics.signals_by_type.get("listening") == 1

    def test_returns_system_metrics(self) -> None:
        self.orchestrator.emit(
            "trace", make_trace(conversation_id=ConversationId("conv-1"))
        )
        self.runtime.emit("message_committed", {"conversation_id": "conv-1", "entity_id": "e-1"})
        self.runtime.emit("message_committed", {"conversation_id": "conv-2", "entity_id": "e-1"})

        sys = self.collector.get_system_metrics()
        assert sys.total_conversations >= 1
        assert sys.total_events >= 2
        assert sys.uptime_ms >= 0

    def test_reset_clears_all_metrics(self) -> None:
        self.orchestrator.emit("trace", make_trace())
        self.runtime.emit("message_committed", {"conversation_id": "conv-1", "entity_id": "e-1"})

        self.collector.reset()

        sys = self.collector.get_system_metrics()
        assert sys.total_conversations == 0
        assert sys.total_events == 0
        assert sys.active_connections == 0

    def test_returns_empty_metrics_for_unknown_conversation(self) -> None:
        metrics = self.collector.get_conversation_metrics(ConversationId("unknown"))
        assert metrics.event_count == 0
        assert metrics.lock_acquisitions == 0

    def test_returns_empty_metrics_for_unknown_entity(self) -> None:
        metrics = self.collector.get_entity_metrics(EntityId("unknown"))
        assert metrics.response_latency_ms.p50 == 0.0
        assert metrics.disconnection_count == 0

    def test_tracks_yield_rate(self) -> None:
        # 2 dispatches
        self.orchestrator.emit(
            "trace",
            make_trace(dispatch_entity=EntityId("e-1"), outcome="dispatched"),
        )
        self.orchestrator.emit(
            "trace",
            make_trace(dispatch_entity=EntityId("e-1"), outcome="dispatched"),
        )
        # 1 gated (yield)
        self.orchestrator.emit(
            "trace",
            make_trace(
                outcome="gated",
                routing=RoutingDecision(
                    eligible_entities=[],
                    selected_entity=EntityId("e-1"),
                    reason="gated",
                    rule_applied="explicit_mention",
                ),
            ),
        )

        metrics = self.collector.get_entity_metrics(EntityId("e-1"))
        assert metrics.yield_rate > 0

    def test_set_queue_depth(self) -> None:
        self.collector.set_queue_depth(5)
        sys = self.collector.get_system_metrics()
        assert sys.artificer_queue_depth == 5

    def test_reset_clears_queue_depth(self) -> None:
        self.collector.set_queue_depth(5)
        self.collector.reset()
        sys = self.collector.get_system_metrics()
        assert sys.artificer_queue_depth == 0

    def test_message_committed_increments_event_count(self) -> None:
        self.runtime.emit("message_committed", {"conversation_id": "conv-x", "entity_id": "e-1"})
        self.runtime.emit("message_committed", {"conversation_id": "conv-x", "entity_id": "e-1"})
        metrics = self.collector.get_conversation_metrics(ConversationId("conv-x"))
        assert metrics.event_count == 2

    def test_percentile_stats_computed(self) -> None:
        # Record dispatch then simulate latency-tracked commits
        for _ in range(5):
            trace = make_trace(
                dispatch_entity=EntityId("e-perc"),
                outcome="dispatched",
                conversation_id=ConversationId("conv-perc"),
            )
            self.orchestrator.emit("trace", trace)
            time.sleep(0.01)
            self.runtime.emit(
                "message_committed",
                {"conversation_id": "conv-perc", "entity_id": "e-perc"},
            )

        metrics = self.collector.get_entity_metrics(EntityId("e-perc"))
        assert metrics.response_latency_ms.p50 > 0
        assert metrics.response_latency_ms.p95 >= metrics.response_latency_ms.p50
        assert metrics.response_latency_ms.p99 >= metrics.response_latency_ms.p95
