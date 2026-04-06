"""Trace Logger -- §17.2 of the IECP specification.

In-memory ring buffer for orchestration traces.
Supports query by conversation, entity, outcome, and time range.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from ..orchestrator.types import OrchestrationTrace
from ..types.entity import EntityId
from ..types.event import ConversationId


# --- Configuration ----------------------------------------------------------


class TraceLoggerConfig(BaseModel):
    max_traces: int = 10_000
    """Maximum traces to keep in the ring buffer."""


DEFAULT_TRACE_LOGGER_CONFIG = TraceLoggerConfig()


# --- Query Filter -----------------------------------------------------------


class TraceQuery(BaseModel):
    conversation_id: ConversationId | None = None
    entity_id: EntityId | None = None
    outcome: Literal["dispatched", "gated", "no_eligible", "suppressed", "error"] | None = None
    since: float | None = None
    limit: int | None = None


# --- Stats ------------------------------------------------------------------


class TraceStats(BaseModel):
    total_traces: int = 0
    outcome_distribution: dict[str, int] = {}
    avg_pipeline_duration_ms: float = 0.0
    routing_rule_distribution: dict[str, int] = {}


# --- TraceLogger ------------------------------------------------------------


class TraceLogger:
    """In-memory ring buffer for orchestration traces."""

    def __init__(self, config: TraceLoggerConfig | dict[str, Any] | None = None) -> None:
        if config is None:
            self._config = DEFAULT_TRACE_LOGGER_CONFIG
        elif isinstance(config, dict):
            self._config = TraceLoggerConfig(**config)
        else:
            self._config = config

        self._buffer: list[OrchestrationTrace] = []
        self._write_index: int = 0
        self._count: int = 0

    def record(self, trace: OrchestrationTrace) -> None:
        """Record an orchestration trace."""
        if self._count < self._config.max_traces:
            self._buffer.append(trace)
            self._count += 1
        else:
            # Ring buffer — overwrite oldest
            self._buffer[self._write_index] = trace

        self._write_index = (self._write_index + 1) % self._config.max_traces

    def query(self, filter: TraceQuery | dict[str, Any] | None = None) -> list[OrchestrationTrace]:
        """Query traces with optional filters."""
        if filter is None:
            filter = TraceQuery()
        elif isinstance(filter, dict):
            filter = TraceQuery(**{k: v for k, v in filter.items() if v is not None})

        results = list(self._buffer)

        if filter.conversation_id is not None:
            results = [t for t in results if t.conversation_id == filter.conversation_id]
        if filter.entity_id is not None:
            results = [t for t in results if t.dispatch_entity == filter.entity_id]
        if filter.outcome is not None:
            results = [t for t in results if t.outcome == filter.outcome]
        if filter.since is not None:
            results = [t for t in results if t.timestamp >= filter.since]

        # Sort by timestamp descending (most recent first)
        results.sort(key=lambda t: t.timestamp, reverse=True)

        if filter.limit is not None:
            results = results[: filter.limit]

        return results

    def get(self, trace_id: str) -> OrchestrationTrace | None:
        """Get trace by ID."""
        for trace in self._buffer:
            if trace.trace_id == trace_id:
                return trace
        return None

    def get_stats(self) -> TraceStats:
        """Get stats summary."""
        traces = list(self._buffer)
        outcome_dist: dict[str, int] = {}
        routing_dist: dict[str, int] = {}
        total_duration = 0.0

        for trace in traces:
            outcome_dist[trace.outcome] = outcome_dist.get(trace.outcome, 0) + 1
            rule = trace.routing.rule_applied
            routing_dist[rule] = routing_dist.get(rule, 0) + 1
            total_duration += trace.duration_ms

        avg_duration = total_duration / len(traces) if traces else 0.0

        return TraceStats(
            total_traces=len(traces),
            outcome_distribution=outcome_dist,
            avg_pipeline_duration_ms=avg_duration,
            routing_rule_distribution=routing_dist,
        )

    def reset(self) -> None:
        """Reset all traces."""
        self._buffer.clear()
        self._write_index = 0
        self._count = 0
