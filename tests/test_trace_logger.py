"""TraceLogger tests -- Phase 10: Observability."""

from __future__ import annotations

import pytest

from iecp_core.observability import TraceLogger, TraceLoggerConfig, TraceQuery
from iecp_core.orchestrator.types import GatingResult, OrchestrationTrace, RoutingDecision
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId


# --- Helpers ----------------------------------------------------------------


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


# --- Tests ------------------------------------------------------------------


class TestTraceLogger:
    def setup_method(self) -> None:
        self.logger = TraceLogger()

    def test_records_and_retrieves_traces(self) -> None:
        trace = make_trace(trace_id="tr-1")
        self.logger.record(trace)
        assert self.logger.get("tr-1") == trace

    def test_returns_none_for_unknown_trace(self) -> None:
        assert self.logger.get("nonexistent") is None

    def test_queries_by_conversation(self) -> None:
        self.logger.record(make_trace(conversation_id=ConversationId("conv-a")))
        self.logger.record(make_trace(conversation_id=ConversationId("conv-b")))
        self.logger.record(make_trace(conversation_id=ConversationId("conv-a")))

        results = self.logger.query(TraceQuery(conversation_id=ConversationId("conv-a")))
        assert len(results) == 2
        for t in results:
            assert t.conversation_id == "conv-a"

    def test_queries_by_entity(self) -> None:
        self.logger.record(make_trace(dispatch_entity=EntityId("e-1")))
        self.logger.record(make_trace(dispatch_entity=EntityId("e-2")))
        self.logger.record(make_trace(dispatch_entity=EntityId("e-1")))

        results = self.logger.query(TraceQuery(entity_id=EntityId("e-1")))
        assert len(results) == 2

    def test_queries_by_outcome(self) -> None:
        self.logger.record(make_trace(outcome="dispatched"))
        self.logger.record(make_trace(outcome="gated"))
        self.logger.record(make_trace(outcome="dispatched"))

        results = self.logger.query(TraceQuery(outcome="gated"))
        assert len(results) == 1
        assert results[0].outcome == "gated"

    def test_queries_by_since_timestamp(self) -> None:
        self.logger.record(make_trace(timestamp=1000.0))
        self.logger.record(make_trace(timestamp=2000.0))
        self.logger.record(make_trace(timestamp=3000.0))

        results = self.logger.query(TraceQuery(since=2000.0))
        assert len(results) == 2

    def test_limits_results(self) -> None:
        for _ in range(10):
            self.logger.record(make_trace())
        results = self.logger.query(TraceQuery(limit=3))
        assert len(results) == 3

    def test_ring_buffer_evicts_oldest_when_full(self) -> None:
        small_logger = TraceLogger(TraceLoggerConfig(max_traces=3))
        small_logger.record(make_trace(trace_id="old-1"))
        small_logger.record(make_trace(trace_id="old-2"))
        small_logger.record(make_trace(trace_id="old-3"))
        small_logger.record(make_trace(trace_id="new-4"))

        assert small_logger.get("old-1") is None
        assert small_logger.get("new-4") is not None
        assert len(small_logger.query(TraceQuery())) == 3

    def test_get_stats_returns_accurate_distribution(self) -> None:
        self.logger.record(
            make_trace(
                outcome="dispatched",
                duration_ms=10.0,
                routing=RoutingDecision(
                    eligible_entities=[],
                    selected_entity=None,
                    reason="",
                    rule_applied="explicit_mention",
                ),
            )
        )
        self.logger.record(
            make_trace(
                outcome="dispatched",
                duration_ms=20.0,
                routing=RoutingDecision(
                    eligible_entities=[],
                    selected_entity=None,
                    reason="",
                    rule_applied="auto_domain_match",
                ),
            )
        )
        self.logger.record(
            make_trace(
                outcome="gated",
                duration_ms=5.0,
                routing=RoutingDecision(
                    eligible_entities=[],
                    selected_entity=None,
                    reason="",
                    rule_applied="explicit_mention",
                ),
            )
        )

        stats = self.logger.get_stats()
        assert stats.total_traces == 3
        assert stats.outcome_distribution.get("dispatched") == 2
        assert stats.outcome_distribution.get("gated") == 1
        assert stats.routing_rule_distribution.get("explicit_mention") == 2
        assert stats.routing_rule_distribution.get("auto_domain_match") == 1
        assert abs(stats.avg_pipeline_duration_ms - 11.67) < 0.5

    def test_reset_clears_all_traces(self) -> None:
        self.logger.record(make_trace())
        self.logger.record(make_trace())
        self.logger.reset()
        assert len(self.logger.query(TraceQuery())) == 0
        assert self.logger.get_stats().total_traces == 0

    def test_empty_query_returns_all_traces(self) -> None:
        for i in range(5):
            self.logger.record(make_trace(trace_id=f"tr-{i}"))
        results = self.logger.query(TraceQuery())
        assert len(results) == 5

    def test_results_sorted_by_timestamp_descending(self) -> None:
        self.logger.record(make_trace(timestamp=1000.0))
        self.logger.record(make_trace(timestamp=3000.0))
        self.logger.record(make_trace(timestamp=2000.0))

        results = self.logger.query(TraceQuery())
        timestamps = [t.timestamp for t in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_ring_buffer_wraps_correctly(self) -> None:
        small_logger = TraceLogger(TraceLoggerConfig(max_traces=3))
        for i in range(6):
            small_logger.record(make_trace(trace_id=f"trace-{i}"))

        results = small_logger.query(TraceQuery())
        assert len(results) == 3
        # Should have the last 3 added
        ids = {t.trace_id for t in results}
        assert "trace-3" in ids
        assert "trace-4" in ids
        assert "trace-5" in ids
