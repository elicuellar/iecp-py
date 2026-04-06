"""RateLimiter tests -- Phase 10: Observability."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from iecp_core.observability import RateLimiter, RateLimiterConfig
from iecp_core.types.entity import EntityId


class TestRateLimiter:
    def setup_method(self) -> None:
        self.limiter = RateLimiter(RateLimiterConfig(default_limit=5, window_ms=10_000))

    def test_allows_requests_within_limit(self) -> None:
        entity_id = EntityId("e-1")
        for _ in range(5):
            result = self.limiter.check(entity_id)
            assert result.allowed is True
            self.limiter.record(entity_id)

    def test_denies_requests_exceeding_limit(self) -> None:
        entity_id = EntityId("e-1")
        for _ in range(5):
            self.limiter.record(entity_id)
        result = self.limiter.check(entity_id)
        assert result.allowed is False
        assert result.remaining == 0

    def test_sliding_window_resets_after_window_elapses(self) -> None:
        entity_id = EntityId("e-1")

        # Mock time: record 5 requests at t=0
        start = 1_000_000.0
        with patch("time.time", return_value=start / 1000.0):
            for _ in range(5):
                self.limiter.record(entity_id)
            assert self.limiter.check(entity_id).allowed is False

        # Advance time past window (10s = 10000ms)
        future = (start + 10_001) / 1000.0
        with patch("time.time", return_value=future):
            result = self.limiter.check(entity_id)
            assert result.allowed is True
            assert result.remaining == 5

    def test_custom_per_entity_limits(self) -> None:
        entity_id = EntityId("e-1")
        self.limiter.set_limit(entity_id, 2)

        self.limiter.record(entity_id)
        self.limiter.record(entity_id)

        assert self.limiter.check(entity_id).allowed is False

    def test_default_limit_for_unknown_entities(self) -> None:
        entity_id = EntityId("unknown")
        result = self.limiter.check(entity_id)
        assert result.allowed is True
        assert result.remaining == 5

    def test_remaining_count_decreases_correctly(self) -> None:
        entity_id = EntityId("e-1")
        self.limiter.record(entity_id)
        self.limiter.record(entity_id)
        result = self.limiter.check(entity_id)
        assert result.allowed is True
        assert result.remaining == 3

    def test_reset_clears_all_state(self) -> None:
        entity_id = EntityId("e-1")
        self.limiter.set_limit(entity_id, 1)
        self.limiter.record(entity_id)
        assert self.limiter.check(entity_id).allowed is False

        self.limiter.reset()
        result = self.limiter.check(entity_id)
        assert result.allowed is True
        # Custom limit also cleared, default (5) applies
        assert result.remaining == 5

    def test_sliding_window_partially_expires_old_requests(self) -> None:
        entity_id = EntityId("e-1")
        start = 1_000_000.0  # ms

        # Record 3 at t=0
        with patch("time.time", return_value=start / 1000.0):
            self.limiter.record(entity_id)
            self.limiter.record(entity_id)
            self.limiter.record(entity_id)

        # Record 2 more at t=6s
        with patch("time.time", return_value=(start + 6_000) / 1000.0):
            self.limiter.record(entity_id)
            self.limiter.record(entity_id)
            # At t=6s: 5 requests within window -> denied
            assert self.limiter.check(entity_id).allowed is False

        # At t=11s: first 3 are outside the 10s window
        with patch("time.time", return_value=(start + 11_000) / 1000.0):
            result = self.limiter.check(entity_id)
            assert result.allowed is True
            assert result.remaining == 3

    def test_reset_at_reflects_oldest_request(self) -> None:
        entity_id = EntityId("e-1")
        start = 2_000_000.0

        with patch("time.time", return_value=start / 1000.0):
            self.limiter.record(entity_id)
            result = self.limiter.check(entity_id)
            # reset_at = start + window_ms = start + 10000
            assert abs(result.reset_at - (start + 10_000)) < 100

    def test_multiple_entities_independent(self) -> None:
        e1 = EntityId("e-1")
        e2 = EntityId("e-2")

        for _ in range(5):
            self.limiter.record(e1)

        assert self.limiter.check(e1).allowed is False
        assert self.limiter.check(e2).allowed is True
