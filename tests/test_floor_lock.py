"""Floor Lock tests -- Phase 3: Mutual Exclusion.

All tests use a FakeTimerProvider for deterministic timer behavior.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from iecp_core.lock import (
    FloorLock,
    FloorLockConfig,
    LockRequest,
    LockReleasedEvent,
    LockState,
)
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId


# -- FakeTimerProvider -------------------------------------------------------


class FakeTimerProvider:
    """Fake timer for deterministic testing."""

    def __init__(self, start_time: float = 1_000_000.0) -> None:
        self._current_time: float = start_time
        self._timers: dict[int, tuple[float, Callable[[], None]]] = {}
        self._next_handle: int = 0

    def set_timeout(self, callback: Callable[[], None], ms: float) -> int:
        handle = self._next_handle
        self._next_handle += 1
        self._timers[handle] = (self._current_time + ms, callback)
        return handle

    def clear_timeout(self, handle: Any) -> None:
        self._timers.pop(handle, None)

    def now(self) -> float:
        return self._current_time

    def advance(self, ms: float) -> None:
        """Advance time by ms, firing any timers that expire."""
        target = self._current_time + ms
        while True:
            earliest: tuple[int, float, Callable[[], None]] | None = None
            for h, (fire_at, cb) in list(self._timers.items()):
                if fire_at <= target and (earliest is None or fire_at < earliest[1]):
                    earliest = (h, fire_at, cb)
            if earliest is None:
                break
            h, fire_at, cb = earliest
            self._current_time = fire_at
            del self._timers[h]
            cb()
        self._current_time = target


# -- Helpers -----------------------------------------------------------------

CONV_A = ConversationId("conv-a")
CONV_B = ConversationId("conv-b")

ENTITY_1 = EntityId("entity-1")
ENTITY_2 = EntityId("entity-2")
ENTITY_3 = EntityId("entity-3")
ENTITY_4 = EntityId("entity-4")
ENTITY_5 = EntityId("entity-5")


def make_request(**overrides: Any) -> LockRequest:
    defaults: dict[str, Any] = {
        "entity_id": ENTITY_1,
        "conversation_id": CONV_A,
        "estimated_ms": 10_000,
        "priority": "default",
    }
    defaults.update(overrides)
    return LockRequest(**defaults)


CUSTOM_CONFIG = FloorLockConfig(
    default_ttl_ms=5_000,
    max_ttl_ms=20_000,
    ttl_multiplier=2.0,
    grace_period_ms=2_000,
)


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def timer() -> FakeTimerProvider:
    return FakeTimerProvider(start_time=1_000_000.0)


@pytest.fixture
def lock(timer: FakeTimerProvider) -> FloorLock:
    fl = FloorLock(timer_provider=timer)
    yield fl
    fl.destroy()


# -- Basic Acquire/Release ---------------------------------------------------


class TestAcquire:
    def test_grants_lock_when_free(self, lock: FloorLock) -> None:
        result = lock.acquire(make_request())

        assert result.granted is True
        assert result.lock is not None
        assert result.lock.holder_id == ENTITY_1
        assert result.lock.conversation_id == CONV_A

    def test_denies_lock_when_held_by_another_entity(self, lock: FloorLock) -> None:
        lock.acquire(make_request(entity_id=ENTITY_1))
        result = lock.acquire(make_request(entity_id=ENTITY_2))

        assert result.granted is False
        assert result.reason == "lock_held"
        assert result.queue_position == 1

    def test_extends_ttl_when_same_entity_reacquires(
        self, lock: FloorLock, timer: FakeTimerProvider
    ) -> None:
        first = lock.acquire(make_request(estimated_ms=10_000))
        assert first.granted is True

        timer.advance(5_000)

        second = lock.acquire(make_request(estimated_ms=20_000))
        assert second.granted is True
        assert second.lock is not None
        assert second.lock.estimated_ms == 20_000
        # New TTL: max(20000 * 1.5, 30000) = 30000 (default_ttl_ms wins)
        assert second.lock.ttl_ms == 30_000
        assert second.lock.expires_at == 1_005_000 + 30_000


# -- Release -----------------------------------------------------------------


class TestRelease:
    def test_releases_lock_on_commit(self, lock: FloorLock) -> None:
        lock.acquire(make_request())
        released = lock.release(CONV_A, ENTITY_1, "commit")

        assert released is True
        assert lock.is_locked(CONV_A) is False

    def test_releases_lock_on_yield(self, lock: FloorLock) -> None:
        lock.acquire(make_request())
        released = lock.release(CONV_A, ENTITY_1, "yield")

        assert released is True
        assert lock.is_locked(CONV_A) is False

    def test_returns_false_when_entity_does_not_hold_lock(
        self, lock: FloorLock
    ) -> None:
        lock.acquire(make_request(entity_id=ENTITY_1))
        released = lock.release(CONV_A, ENTITY_2, "commit")

        assert released is False
        assert lock.is_locked(CONV_A) is True

    def test_returns_false_when_no_lock_exists(self, lock: FloorLock) -> None:
        released = lock.release(CONV_A, ENTITY_1, "commit")
        assert released is False


# -- TTL Expiry --------------------------------------------------------------


class TestTtlExpiry:
    def test_auto_releases_after_ttl_plus_grace(
        self, lock: FloorLock, timer: FakeTimerProvider
    ) -> None:
        released_events: list[LockReleasedEvent] = []
        lock.on("lock_released", lambda e: released_events.append(e))

        # estimated_ms=10000, TTL = max(10000*1.5, 30000) = 30000
        lock.acquire(make_request(estimated_ms=10_000))

        # Advance past TTL (30s)
        timer.advance(30_000)
        assert lock.is_locked(CONV_A) is True  # Still in grace period

        # Advance past grace period (5s default)
        timer.advance(5_000)
        assert lock.is_locked(CONV_A) is False
        assert len(released_events) == 1
        assert released_events[0].reason == "ttl_expired"

    def test_allows_late_commit_during_grace_period(
        self, lock: FloorLock, timer: FakeTimerProvider
    ) -> None:
        released_events: list[LockReleasedEvent] = []
        lock.on("lock_released", lambda e: released_events.append(e))

        lock.acquire(make_request(estimated_ms=10_000))

        # Advance past TTL but within grace period
        timer.advance(30_000 + 2_000)  # 2s into grace
        assert lock.is_locked(CONV_A) is True

        # Entity commits during grace
        lock.release(CONV_A, ENTITY_1, "commit")
        assert lock.is_locked(CONV_A) is False
        assert len(released_events) == 1
        assert released_events[0].reason == "commit"

    def test_force_releases_when_entity_does_not_commit_during_grace(
        self, lock: FloorLock, timer: FakeTimerProvider
    ) -> None:
        released_events: list[LockReleasedEvent] = []
        lock.on("lock_released", lambda e: released_events.append(e))

        lock.acquire(make_request(estimated_ms=10_000))

        # Advance past TTL (30s) + full grace period (5s)
        timer.advance(35_000)
        assert lock.is_locked(CONV_A) is False
        assert len(released_events) == 1
        assert released_events[0].reason == "ttl_expired"


# -- Human Interruption ------------------------------------------------------


class TestHandleHumanInterrupt:
    def test_immediately_releases_lock_and_clears_queue(
        self, lock: FloorLock
    ) -> None:
        released_events: list[LockReleasedEvent] = []
        lock.on("lock_released", lambda e: released_events.append(e))

        lock.acquire(make_request(entity_id=ENTITY_1))
        lock.acquire(make_request(entity_id=ENTITY_2))
        lock.acquire(make_request(entity_id=ENTITY_3))

        assert lock.get_queue_length(CONV_A) == 2

        result = lock.handle_human_interrupt(CONV_A)

        assert result is True
        assert lock.is_locked(CONV_A) is False
        assert lock.get_queue_length(CONV_A) == 0
        assert len(released_events) == 1
        assert released_events[0].reason == "human_interrupt"

    def test_returns_false_when_no_lock_is_held(self, lock: FloorLock) -> None:
        assert lock.handle_human_interrupt(CONV_A) is False


# -- Tie-Breaking ------------------------------------------------------------


class TestTieBreaking:
    def test_mention_priority_beats_artificer(self, lock: FloorLock) -> None:
        lock.acquire(make_request(entity_id=ENTITY_1))

        # Both queue up
        lock.acquire(make_request(entity_id=ENTITY_2, priority="artificer"))
        lock.acquire(make_request(entity_id=ENTITY_3, priority="mention"))

        # Release -- mention priority should win
        lock.release(CONV_A, ENTITY_1, "commit")

        assert lock.is_locked(CONV_A) is True
        assert lock.get_lock_state(CONV_A).holder_id == ENTITY_3

    def test_artificer_priority_beats_daemon(self, lock: FloorLock) -> None:
        lock.acquire(make_request(entity_id=ENTITY_1))

        lock.acquire(make_request(entity_id=ENTITY_2, priority="daemon"))
        lock.acquire(make_request(entity_id=ENTITY_3, priority="artificer"))

        lock.release(CONV_A, ENTITY_1, "commit")

        assert lock.get_lock_state(CONV_A).holder_id == ENTITY_3

    def test_same_priority_uses_round_robin(self, lock: FloorLock) -> None:
        # First cycle: ENTITY_1 gets the lock
        lock.acquire(make_request(entity_id=ENTITY_1, priority="default"))

        # ENTITY_2 and ENTITY_3 queue up
        lock.acquire(make_request(entity_id=ENTITY_2, priority="default"))
        lock.acquire(make_request(entity_id=ENTITY_3, priority="default"))

        # Release ENTITY_1 -- ENTITY_2 goes next (FIFO, ENTITY_1 was lastServed)
        lock.release(CONV_A, ENTITY_1, "commit")
        assert lock.get_lock_state(CONV_A).holder_id == ENTITY_2

        # Now ENTITY_1 queues up again
        lock.acquire(make_request(entity_id=ENTITY_1, priority="default"))

        # Release ENTITY_2 -- ENTITY_3 goes next (round-robin: ENTITY_2 was lastServed)
        lock.release(CONV_A, ENTITY_2, "commit")
        assert lock.get_lock_state(CONV_A).holder_id == ENTITY_3


# -- Queue Behavior ----------------------------------------------------------


class TestQueue:
    def test_auto_grants_to_next_in_queue_on_release(
        self, lock: FloorLock
    ) -> None:
        acquired_events: list[LockState] = []
        lock.on("lock_acquired", lambda s: acquired_events.append(s))

        lock.acquire(make_request(entity_id=ENTITY_1))
        lock.acquire(make_request(entity_id=ENTITY_2))

        lock.release(CONV_A, ENTITY_1, "commit")

        assert lock.get_lock_state(CONV_A).holder_id == ENTITY_2
        # 2 acquired events: initial grant + auto-grant
        assert len(acquired_events) == 2

    def test_expires_queue_entries_after_max_ttl_ms(
        self, timer: FakeTimerProvider
    ) -> None:
        # Use a short TTL config so lock doesn't auto-expire before queue entries
        lock = FloorLock(
            config=FloorLockConfig(
                default_ttl_ms=5_000,
                max_ttl_ms=10_000,
                grace_period_ms=2_000,
            ),
            timer_provider=timer,
        )

        lock.acquire(make_request(entity_id=ENTITY_1, estimated_ms=5_000))
        lock.acquire(make_request(entity_id=ENTITY_2, estimated_ms=5_000))

        assert lock.get_queue_length(CONV_A) == 1

        # Keep extending the lock to prevent TTL expiry, while time passes
        # Queue entry expires at now + max_ttl_ms = now + 10000
        timer.advance(4_000)
        lock.acquire(make_request(entity_id=ENTITY_1, estimated_ms=5_000))  # extend
        timer.advance(4_000)
        lock.acquire(make_request(entity_id=ENTITY_1, estimated_ms=5_000))  # extend
        timer.advance(3_000)
        # Now 11s have passed -- queue entry has expired

        # Release ENTITY_1's lock
        lock.release(CONV_A, ENTITY_1, "commit")

        # ENTITY_2's queue entry expired -- no one gets the lock
        assert lock.is_locked(CONV_A) is False

        lock.destroy()

    def test_allows_entity_to_cancel_queue_position(
        self, lock: FloorLock
    ) -> None:
        lock.acquire(make_request(entity_id=ENTITY_1))
        lock.acquire(make_request(entity_id=ENTITY_2))
        lock.acquire(make_request(entity_id=ENTITY_3))

        assert lock.get_queue_length(CONV_A) == 2

        cancelled = lock.cancel_queue(CONV_A, ENTITY_2)
        assert cancelled is True
        assert lock.get_queue_length(CONV_A) == 1

        # Release -- ENTITY_3 should get it
        lock.release(CONV_A, ENTITY_1, "commit")
        assert lock.get_lock_state(CONV_A).holder_id == ENTITY_3

    def test_returns_false_for_cancel_when_entity_not_in_queue(
        self, lock: FloorLock
    ) -> None:
        assert lock.cancel_queue(CONV_A, ENTITY_1) is False


# -- Concurrent Requests -----------------------------------------------------


class TestConcurrentRequests:
    def test_grants_exactly_1_of_5_requests_queues_4(
        self, lock: FloorLock
    ) -> None:
        entities = [ENTITY_1, ENTITY_2, ENTITY_3, ENTITY_4, ENTITY_5]
        results = [lock.acquire(make_request(entity_id=eid)) for eid in entities]

        granted = [r for r in results if r.granted]
        denied = [r for r in results if not r.granted]

        assert len(granted) == 1
        assert len(denied) == 4
        assert granted[0].lock.holder_id == ENTITY_1

        # All denied should have queue positions 1-4
        positions = sorted(r.queue_position for r in denied)
        assert positions == [1, 2, 3, 4]

    def test_drains_queue_sequentially_on_repeated_releases(
        self, lock: FloorLock
    ) -> None:
        entities = [ENTITY_1, ENTITY_2, ENTITY_3, ENTITY_4, ENTITY_5]
        for eid in entities:
            lock.acquire(make_request(entity_id=eid))

        # Each entity should hold the lock in turn
        for eid in entities:
            assert lock.get_lock_state(CONV_A).holder_id == eid
            lock.release(CONV_A, eid, "commit")

        assert lock.is_locked(CONV_A) is False
        assert lock.get_queue_length(CONV_A) == 0


# -- State Queries -----------------------------------------------------------


class TestStateQueries:
    def test_get_lock_state_returns_none_when_not_locked(
        self, lock: FloorLock
    ) -> None:
        assert lock.get_lock_state(CONV_A) is None

    def test_get_lock_state_returns_lock_when_held(
        self, lock: FloorLock
    ) -> None:
        lock.acquire(make_request())
        state = lock.get_lock_state(CONV_A)
        assert state is not None
        assert state.holder_id == ENTITY_1

    def test_is_locked_returns_correct_state(self, lock: FloorLock) -> None:
        assert lock.is_locked(CONV_A) is False
        lock.acquire(make_request())
        assert lock.is_locked(CONV_A) is True

    def test_get_queue_position_returns_none_when_not_in_queue(
        self, lock: FloorLock
    ) -> None:
        assert lock.get_queue_position(CONV_A, ENTITY_1) is None

    def test_get_queue_position_returns_1_indexed_position(
        self, lock: FloorLock
    ) -> None:
        lock.acquire(make_request(entity_id=ENTITY_1))
        lock.acquire(make_request(entity_id=ENTITY_2))
        lock.acquire(make_request(entity_id=ENTITY_3))

        assert lock.get_queue_position(CONV_A, ENTITY_2) == 1
        assert lock.get_queue_position(CONV_A, ENTITY_3) == 2

    def test_get_queue_length_returns_0_for_unknown_conversations(
        self, lock: FloorLock
    ) -> None:
        assert lock.get_queue_length(CONV_B) == 0


# -- Destroy -----------------------------------------------------------------


class TestDestroy:
    def test_clears_all_timers_and_state(
        self, timer: FakeTimerProvider
    ) -> None:
        lock = FloorLock(timer_provider=timer)
        lock.acquire(make_request(conversation_id=CONV_A))
        lock.acquire(make_request(conversation_id=CONV_B, entity_id=ENTITY_2))

        lock.destroy()

        assert lock.is_locked(CONV_A) is False
        assert lock.is_locked(CONV_B) is False

        # Advancing timers should not throw or trigger events
        timer.advance(100_000)


# -- Configuration -----------------------------------------------------------


class TestConfiguration:
    def test_respects_custom_ttl_values(self, timer: FakeTimerProvider) -> None:
        lock = FloorLock(config=CUSTOM_CONFIG, timer_provider=timer)

        result = lock.acquire(make_request(estimated_ms=3_000))
        # TTL = max(3000 * 2.0, 5000) = 6000, capped at 20000 -> 6000
        assert result.lock.ttl_ms == 6_000

        lock.destroy()

    def test_applies_ttl_multiplier_correctly(
        self, timer: FakeTimerProvider
    ) -> None:
        lock = FloorLock(config=CUSTOM_CONFIG, timer_provider=timer)

        result = lock.acquire(make_request(estimated_ms=8_000))
        # TTL = max(8000 * 2.0, 5000) = 16000, capped at 20000 -> 16000
        assert result.lock.ttl_ms == 16_000

        lock.destroy()

    def test_caps_ttl_at_max_ttl_ms(self, timer: FakeTimerProvider) -> None:
        lock = FloorLock(config=CUSTOM_CONFIG, timer_provider=timer)

        result = lock.acquire(make_request(estimated_ms=15_000))
        # TTL = max(15000 * 2.0, 5000) = 30000, capped at 20000 -> 20000
        assert result.lock.ttl_ms == 20_000

        lock.destroy()

    def test_uses_default_ttl_ms_as_floor(self, lock: FloorLock) -> None:
        # Default config: default_ttl_ms=30000, multiplier=1.5
        result = lock.acquire(make_request(estimated_ms=1_000))
        # TTL = max(1000 * 1.5, 30000) = 30000
        assert result.lock.ttl_ms == 30_000

    def test_custom_grace_period_is_respected(
        self, timer: FakeTimerProvider
    ) -> None:
        lock = FloorLock(config=CUSTOM_CONFIG, timer_provider=timer)

        released_events: list[LockReleasedEvent] = []
        lock.on("lock_released", lambda e: released_events.append(e))

        lock.acquire(make_request(estimated_ms=3_000))
        # TTL = 6000ms

        # Advance past TTL
        timer.advance(6_000)
        assert lock.is_locked(CONV_A) is True  # In grace period

        # Advance 1s into 2s grace -> still locked
        timer.advance(1_000)
        assert lock.is_locked(CONV_A) is True

        # Advance past the remaining grace
        timer.advance(1_000)
        assert lock.is_locked(CONV_A) is False
        assert released_events[0].reason == "ttl_expired"

        lock.destroy()


# -- Event Emission ----------------------------------------------------------


class TestEvents:
    def test_emits_lock_acquired_on_grant(self, lock: FloorLock) -> None:
        acquired_events: list[LockState] = []
        lock.on("lock_acquired", lambda s: acquired_events.append(s))

        lock.acquire(make_request())

        assert len(acquired_events) == 1
        assert acquired_events[0].holder_id == ENTITY_1

    def test_does_not_emit_lock_acquired_on_extend(
        self, lock: FloorLock
    ) -> None:
        acquired_events: list[LockState] = []
        lock.on("lock_acquired", lambda s: acquired_events.append(s))

        lock.acquire(make_request())
        lock.acquire(make_request())  # Same entity re-acquires

        assert len(acquired_events) == 1

    def test_emits_lock_released_on_release(self, lock: FloorLock) -> None:
        released_events: list[LockReleasedEvent] = []
        lock.on("lock_released", lambda e: released_events.append(e))

        lock.acquire(make_request())
        lock.release(CONV_A, ENTITY_1, "commit")

        assert len(released_events) == 1
        assert released_events[0].reason == "commit"
        assert released_events[0].state.holder_id == ENTITY_1

    def test_listener_can_be_removed_with_off(self, lock: FloorLock) -> None:
        acquired_events: list[LockState] = []
        listener = lambda s: acquired_events.append(s)
        lock.on("lock_acquired", listener)
        lock.off("lock_acquired", listener)

        lock.acquire(make_request())

        assert len(acquired_events) == 0


# -- Multi-Conversation Isolation --------------------------------------------


class TestMultiConversationIsolation:
    def test_locks_on_different_conversations_are_independent(
        self, lock: FloorLock
    ) -> None:
        lock.acquire(make_request(conversation_id=CONV_A, entity_id=ENTITY_1))
        result = lock.acquire(
            make_request(conversation_id=CONV_B, entity_id=ENTITY_2)
        )

        assert result.granted is True
        assert lock.is_locked(CONV_A) is True
        assert lock.is_locked(CONV_B) is True
        assert lock.get_lock_state(CONV_A).holder_id == ENTITY_1
        assert lock.get_lock_state(CONV_B).holder_id == ENTITY_2


# -- Metadata ----------------------------------------------------------------


class TestMetadata:
    def test_stores_metadata_on_the_lock(self, lock: FloorLock) -> None:
        result = lock.acquire(make_request(metadata={"intent": "code-review"}))
        assert result.lock.metadata == {"intent": "code-review"}

    def test_merges_metadata_on_extend(self, lock: FloorLock) -> None:
        lock.acquire(make_request(metadata={"intent": "code-review"}))
        result = lock.acquire(make_request(metadata={"extended": True}))

        assert result.lock.metadata == {
            "intent": "code-review",
            "extended": True,
        }

    def test_defaults_to_empty_object_when_no_metadata_provided(
        self, lock: FloorLock
    ) -> None:
        result = lock.acquire(make_request())
        assert result.lock.metadata == {}
