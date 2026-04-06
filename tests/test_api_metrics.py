"""Metrics API tests -- Phase 10."""

from __future__ import annotations

import pytest

from iecp_core.observability import MetricsCollector, TraceLogger
from iecp_core.orchestrator.types import GatingResult, OrchestrationTrace, RoutingDecision
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId

from .api_helpers import auth_headers, create_test_app


def make_trace(**overrides) -> OrchestrationTrace:
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
    return OrchestrationTrace(
        trace_id=overrides.pop("trace_id", f"trace-{id(object())}"),
        conversation_id=overrides.pop("conversation_id", ConversationId("conv-1")),
        timestamp=overrides.pop("timestamp", 1000.0),
        routing=routing,
        gating=gating,
        outcome=overrides.pop("outcome", "dispatched"),
        duration_ms=overrides.pop("duration_ms", 10.0),
        dispatch_entity=overrides.pop("dispatch_entity", EntityId("entity-1")),
        **overrides,
    )


class TestMetricsRoutes:
    def setup_method(self) -> None:
        self.metrics_collector = MetricsCollector()
        self.trace_logger = TraceLogger()
        self.client, self.services = create_test_app(
            metrics_collector=self.metrics_collector,
            trace_logger=self.trace_logger,
        )

    def test_get_system_metrics(self) -> None:
        res = self.client.get("/api/v1/metrics", headers=auth_headers())
        assert res.status_code == 200
        body = res.json()
        assert "total_conversations" in body
        assert "total_events" in body
        assert "active_connections" in body
        assert "artificer_queue_depth" in body
        assert "uptime_ms" in body

    def test_get_conversation_metrics(self) -> None:
        self.metrics_collector.record_decision_proposed(ConversationId("conv-1"))
        self.metrics_collector.record_handoff(ConversationId("conv-1"))

        res = self.client.get(
            "/api/v1/metrics/conversations/conv-1",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        body = res.json()
        assert body["decisions_proposed"] == 1
        assert body["handoff_count"] == 1
        assert "event_count" in body
        assert "lock_acquisitions" in body

    def test_get_entity_metrics(self) -> None:
        self.metrics_collector.record_signal(EntityId("e-1"), "thinking")

        res = self.client.get(
            "/api/v1/metrics/entities/e-1",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        body = res.json()
        assert "response_latency_ms" in body
        assert body["signals_by_type"]["thinking"] == 1

    def test_get_traces(self) -> None:
        self.trace_logger.record(make_trace(trace_id="tr-1", outcome="dispatched"))
        self.trace_logger.record(make_trace(trace_id="tr-2", outcome="gated"))

        res = self.client.get("/api/v1/metrics/traces", headers=auth_headers())
        assert res.status_code == 200
        body = res.json()
        assert len(body["traces"]) == 2
        assert body["stats"]["total_traces"] == 2

    def test_get_traces_filters_by_outcome(self) -> None:
        self.trace_logger.record(make_trace(outcome="dispatched"))
        self.trace_logger.record(make_trace(outcome="gated"))

        res = self.client.get(
            "/api/v1/metrics/traces?outcome=gated",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        body = res.json()
        assert len(body["traces"]) == 1
        assert body["traces"][0]["outcome"] == "gated"

    def test_get_single_trace(self) -> None:
        self.trace_logger.record(make_trace(trace_id="tr-abc"))

        res = self.client.get(
            "/api/v1/metrics/traces/tr-abc",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        assert res.json()["trace_id"] == "tr-abc"

    def test_get_trace_returns_404_for_unknown(self) -> None:
        res = self.client.get(
            "/api/v1/metrics/traces/nonexistent",
            headers=auth_headers(),
        )
        assert res.status_code == 404

    def test_metrics_routes_not_available_without_collector(self) -> None:
        # App without metrics collector should not have the metrics routes
        client_no_metrics, _ = create_test_app()
        res = client_no_metrics.get("/api/v1/metrics", headers=auth_headers())
        assert res.status_code == 404

    def test_health_returns_enhanced_readiness_check(self) -> None:
        res = self.client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert "uptime" in body
        assert "checks" in body
        assert body["checks"]["database"] == "ok"
        assert body["checks"]["gateway"] == "ok"
        assert "artificerQueue" in body["checks"]
        assert body["version"] == "1.0.0-rc1"

    def test_get_traces_with_limit(self) -> None:
        for i in range(5):
            self.trace_logger.record(make_trace(trace_id=f"tr-{i}"))

        res = self.client.get(
            "/api/v1/metrics/traces?limit=3",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        assert len(res.json()["traces"]) == 3

    def test_get_traces_filters_by_conversation(self) -> None:
        self.trace_logger.record(make_trace(conversation_id=ConversationId("conv-a")))
        self.trace_logger.record(make_trace(conversation_id=ConversationId("conv-b")))
        self.trace_logger.record(make_trace(conversation_id=ConversationId("conv-a")))

        res = self.client.get(
            "/api/v1/metrics/traces?conversationId=conv-a",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        assert len(res.json()["traces"]) == 2
