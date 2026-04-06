"""DaemonBuffer tests -- Phase 7."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from iecp_core.gateway import DaemonBuffer
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId, Event, EventId, MessageContent


# -- Helpers -----------------------------------------------------------------

DAEMON_ID = EntityId("daemon-1")


def make_event(event_id: str) -> Event:
    return Event(
        id=EventId(event_id),
        conversation_id=ConversationId("conv-1"),
        author_id=EntityId("author-1"),
        author_type="human",
        type="message",
        content=MessageContent(text=f"Message {event_id}", format="plain", mentions=[]),
        is_continuation=False,
        is_complete=True,
        ai_depth_counter=0,
        status="active",
        metadata={},
        created_at=datetime.now(timezone.utc).isoformat(),
    )


class FakeClock:
    """A controllable clock for testing TTL behaviour."""

    def __init__(self, start_ms: float = 1_000_000.0) -> None:
        self._time_ms = start_ms

    def advance(self, ms: float) -> None:
        self._time_ms += ms

    def time(self) -> float:
        """Returns seconds (like time.time())."""
        return self._time_ms / 1000.0


# -- Tests -------------------------------------------------------------------


class TestBufferFlush:
    def test_buffers_events_for_a_disconnected_daemon(self) -> None:
        buf = DaemonBuffer(ttl_ms=5000, max_events=5)
        buf.buffer(DAEMON_ID, make_event("e1"))
        buf.buffer(DAEMON_ID, make_event("e2"))

        assert buf.has_events(DAEMON_ID)
        assert buf.get_buffer_size(DAEMON_ID) == 2

    def test_flush_returns_events_in_order_and_clears_buffer(self) -> None:
        buf = DaemonBuffer(ttl_ms=5000, max_events=5)
        buf.buffer(DAEMON_ID, make_event("e1"))
        buf.buffer(DAEMON_ID, make_event("e2"))
        buf.buffer(DAEMON_ID, make_event("e3"))

        flushed = buf.flush(DAEMON_ID)
        assert len(flushed) == 3
        assert flushed[0].id == EventId("e1")
        assert flushed[1].id == EventId("e2")
        assert flushed[2].id == EventId("e3")

        # Buffer should be empty after flush
        assert not buf.has_events(DAEMON_ID)
        assert buf.get_buffer_size(DAEMON_ID) == 0

    def test_flush_returns_empty_list_for_unknown_daemon(self) -> None:
        buf = DaemonBuffer(ttl_ms=5000, max_events=5)
        assert buf.flush(EntityId("unknown")) == []


class TestTTLExpiry:
    def test_clear_expired_removes_buffers_past_ttl(self) -> None:
        clock = FakeClock()
        with patch("iecp_core.gateway.daemon_buffer.time") as mock_time:
            mock_time.time.side_effect = clock.time

            buf = DaemonBuffer(ttl_ms=5000, max_events=5)
            buf.buffer(DAEMON_ID, make_event("e1"))

            # Advance past TTL
            clock.advance(6000)

            expired = buf.clear_expired()
            assert DAEMON_ID in expired
            assert not buf.has_events(DAEMON_ID)

    def test_clear_expired_keeps_buffers_within_ttl(self) -> None:
        clock = FakeClock()
        with patch("iecp_core.gateway.daemon_buffer.time") as mock_time:
            mock_time.time.side_effect = clock.time

            buf = DaemonBuffer(ttl_ms=5000, max_events=5)
            buf.buffer(DAEMON_ID, make_event("e1"))

            # Advance but NOT past TTL
            clock.advance(3000)

            expired = buf.clear_expired()
            assert len(expired) == 0
            assert buf.has_events(DAEMON_ID)

    def test_clear_expired_only_removes_expired_keeps_fresh(self) -> None:
        clock = FakeClock()
        old_daemon = EntityId("daemon-old")
        fresh_daemon = EntityId("daemon-fresh")

        with patch("iecp_core.gateway.daemon_buffer.time") as mock_time:
            mock_time.time.side_effect = clock.time

            buf = DaemonBuffer(ttl_ms=5000, max_events=5)
            buf.buffer(old_daemon, make_event("e1"))

            clock.advance(4000)
            buf.buffer(fresh_daemon, make_event("e2"))

            clock.advance(2000)  # old = 6s total, fresh = 2s

            expired = buf.clear_expired()
            assert old_daemon in expired
            assert fresh_daemon not in expired
            assert buf.has_events(fresh_daemon)


class TestMaxEventsLimit:
    def test_drops_oldest_events_when_max_exceeded(self) -> None:
        buf = DaemonBuffer(ttl_ms=5000, max_events=5)

        for i in range(1, 9):
            buf.buffer(DAEMON_ID, make_event(f"e{i}"))

        assert buf.get_buffer_size(DAEMON_ID) == 5

        flushed = buf.flush(DAEMON_ID)
        # Should have kept e4-e8 (dropped e1-e3)
        assert flushed[0].id == EventId("e4")
        assert flushed[4].id == EventId("e8")


class TestHasEventsGetBufferSize:
    def test_has_events_returns_false_for_empty_unknown(self) -> None:
        buf = DaemonBuffer(ttl_ms=5000, max_events=5)
        assert not buf.has_events(DAEMON_ID)

    def test_get_buffer_size_returns_0_for_unknown(self) -> None:
        buf = DaemonBuffer(ttl_ms=5000, max_events=5)
        assert buf.get_buffer_size(EntityId("unknown")) == 0


class TestClearBuffer:
    def test_clears_buffer_for_a_specific_daemon(self) -> None:
        buf = DaemonBuffer(ttl_ms=5000, max_events=5)
        buf.buffer(DAEMON_ID, make_event("e1"))
        buf.clear_buffer(DAEMON_ID)
        assert not buf.has_events(DAEMON_ID)


class TestDestroy:
    def test_clears_all_buffers(self) -> None:
        buf = DaemonBuffer(ttl_ms=5000, max_events=5)
        buf.buffer(DAEMON_ID, make_event("e1"))
        buf.buffer(EntityId("d2"), make_event("e2"))
        buf.destroy()
        assert not buf.has_events(DAEMON_ID)
        assert not buf.has_events(EntityId("d2"))
