"""Debouncer tests -- Phase 2: Smart Debouncing & Batch Sealing.

All tests use a FakeTimerProvider for deterministic timer behavior.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from iecp_core.debounce import (
    DEFAULT_DEBOUNCER_CONFIG,
    Debouncer,
    DebouncerConfig,
    SealedBatch,
)
from iecp_core.events.event_factory import create_message_event
from iecp_core.types.event import ConversationId, Event, EventId


# -- FakeTimerProvider -------------------------------------------------------


class FakeTimerProvider:
    """Fake timer for deterministic testing."""

    def __init__(self) -> None:
        self._current_time: float = 0.0
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
            # Find earliest timer that fires <= target
            earliest: tuple[int, float, Callable[[], None]] | None = None
            for h, (fire_at, cb) in list(self._timers.items()):
                if fire_at <= target and (earliest is None or fire_at < earliest[1]):
                    earliest = (h, fire_at, cb)
            if earliest is None:
                break
            h, fire_at, cb = earliest
            self._current_time = fire_at
            del self._timers[h]
            cb()  # callback may add new timers
        self._current_time = target


# -- Helpers -----------------------------------------------------------------

CONV_A = ConversationId("conv-a")
CONV_B = ConversationId("conv-b")
AUTHOR_A = "author-a"
AUTHOR_B = "author-b"


def make_message(
    conversation_id: ConversationId | None = None,
    author_id: str | None = None,
    is_continuation: bool = False,
    text: str = "hello",
) -> Event:
    return create_message_event(
        conversation_id=conversation_id or CONV_A,
        author_id=author_id or AUTHOR_A,
        author_type="human",
        text=text,
        is_continuation=is_continuation,
    )


def collect_batches(debouncer: Debouncer) -> list[SealedBatch]:
    batches: list[SealedBatch] = []
    debouncer.on("batch_sealed", lambda b: batches.append(b))
    return batches


BASE_CONFIG = DebouncerConfig(
    base_ms=3000,
    min_ms=1500,
    max_ms=8000,
    adaptive=False,  # disabled by default in tests for predictability
    history_window=20,
)


# -- Tests -------------------------------------------------------------------


class TestDebouncerSingleMessage:
    """Single message -> timer fires -> batch sealed with 1 event."""

    def test_seals_a_single_message_after_the_timer_fires(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        msg = make_message()
        debouncer.handle_event(msg)

        assert len(batches) == 0
        timer.advance(3000)
        assert len(batches) == 1
        assert batches[0].event_ids == [msg.id]
        assert batches[0].message_count == 1
        assert batches[0].conversation_id == CONV_A
        assert batches[0].author_id == AUTHOR_A

        debouncer.destroy()


class TestDebouncerMultipleRapidMessages:
    """Multiple rapid messages -> all grouped in one batch."""

    def test_groups_multiple_rapid_messages_into_a_single_batch(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        msg1 = make_message(text="oh wait")
        msg2 = make_message(text="I just realized")
        msg3 = make_message(text="the auth is broken")

        debouncer.handle_event(msg1)
        timer.advance(500)
        debouncer.handle_event(msg2)
        timer.advance(500)
        debouncer.handle_event(msg3)

        assert len(batches) == 0
        timer.advance(3000)
        assert len(batches) == 1
        assert batches[0].event_ids == [msg1.id, msg2.id, msg3.id]
        assert batches[0].message_count == 3

        debouncer.destroy()


class TestDebouncerTimerReset:
    """Timer resets on each new message."""

    def test_resets_the_timer_on_each_new_message(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        debouncer.handle_event(make_message(text="first"))
        timer.advance(2500)  # 500ms left on original timer
        debouncer.handle_event(make_message(text="second"))
        timer.advance(2500)  # Timer was reset, 500ms left again

        assert len(batches) == 0  # Not sealed yet

        timer.advance(500)  # Now 3000ms since last message
        assert len(batches) == 1
        assert batches[0].message_count == 2

        debouncer.destroy()


class TestDebouncerEventOrder:
    """Batch sealed contains correct event_ids in order."""

    def test_preserves_event_id_order_in_sealed_batch(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        events: list[Event] = []
        for i in range(5):
            msg = make_message(text=f"msg-{i}")
            events.append(msg)
            debouncer.handle_event(msg)
            timer.advance(100)

        timer.advance(3000)
        assert batches[0].event_ids == [e.id for e in events]

        debouncer.destroy()


class TestDebouncerIndependentAuthors:
    """Two authors -> independent batches."""

    def test_creates_independent_batches_for_different_authors(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        msg_a = make_message(author_id=AUTHOR_A, text="from A")
        msg_b = make_message(author_id=AUTHOR_B, text="from B")

        debouncer.handle_event(msg_a)
        timer.advance(500)
        debouncer.handle_event(msg_b)

        timer.advance(2500)  # A's timer fires (3000ms since A's message)
        assert len(batches) == 1
        assert batches[0].author_id == AUTHOR_A
        assert batches[0].event_ids == [msg_a.id]

        timer.advance(500)  # B's timer fires (3000ms since B's message)
        assert len(batches) == 2
        assert batches[1].author_id == AUTHOR_B
        assert batches[1].event_ids == [msg_b.id]

        debouncer.destroy()


class TestDebouncerIndependentConversations:
    """Two conversations -> independent batches."""

    def test_creates_independent_batches_for_different_conversations(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        msg_conv_a = make_message(conversation_id=CONV_A, text="conv-a")
        msg_conv_b = make_message(conversation_id=CONV_B, text="conv-b")

        debouncer.handle_event(msg_conv_a)
        debouncer.handle_event(msg_conv_b)

        timer.advance(3000)
        assert len(batches) == 2

        conv_ids = sorted([b.conversation_id for b in batches])
        assert conv_ids == sorted([CONV_A, CONV_B])

        debouncer.destroy()


class TestDebouncerContinuation:
    """is_continuation logic."""

    def test_does_not_seal_batch_when_last_message_has_is_continuation(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        debouncer.handle_event(make_message(text="start"))
        timer.advance(1000)
        debouncer.handle_event(
            make_message(text="still typing...", is_continuation=True)
        )

        timer.advance(3000)  # Timer fires but is_continuation is true
        assert len(batches) == 0

        # Timer was re-extended
        timer.advance(3000)  # Still continuation, extends again
        assert len(batches) == 0

        debouncer.destroy()

    def test_seals_after_normal_message_following_a_continuation(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        msg1 = make_message(text="start")
        debouncer.handle_event(msg1)
        timer.advance(500)

        msg2 = make_message(text="still going...", is_continuation=True)
        debouncer.handle_event(msg2)
        timer.advance(3000)  # Timer fires, extends due to continuation
        assert len(batches) == 0

        # Now send a non-continuation message -- resets the timer
        msg3 = make_message(text="done now")
        debouncer.handle_event(msg3)
        timer.advance(3000)

        assert len(batches) == 1
        assert batches[0].event_ids == [msg1.id, msg2.id, msg3.id]

        debouncer.destroy()

    def test_never_seals_if_all_messages_are_continuations(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        debouncer.handle_event(make_message(text="a", is_continuation=True))
        timer.advance(3000)  # extends
        debouncer.handle_event(make_message(text="b", is_continuation=True))
        timer.advance(3000)  # extends again
        timer.advance(3000)  # extends again

        assert len(batches) == 0

        debouncer.destroy()


class TestDebouncerTypingStart:
    """typing_start handling."""

    def test_resets_the_timer_on_typing_start(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        debouncer.handle_event(make_message(text="hello"))
        timer.advance(2500)  # 500ms left

        debouncer.handle_typing_start(CONV_A, AUTHOR_A)
        timer.advance(2500)  # Timer was reset, 500ms left

        assert len(batches) == 0

        timer.advance(500)  # Now 3000ms since typing_start
        assert len(batches) == 1

        debouncer.destroy()

    def test_ignores_typing_start_when_no_active_batch(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        # Typing start without any messages -- should not throw or create a batch
        debouncer.handle_typing_start(CONV_A, AUTHOR_A)

        timer.advance(10000)
        assert len(batches) == 0

        debouncer.destroy()


class TestDebouncerFiltering:
    """Event type and author type filtering."""

    def test_ignores_non_message_event_types(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        # Create an attention event (not a message)
        from iecp_core.events.event_factory import create_attention_event

        event = create_attention_event(
            conversation_id=CONV_A,
            author_id=AUTHOR_A,
            author_type="human",
            signal="ping",
        )

        debouncer.handle_event(event)
        timer.advance(5000)
        assert len(batches) == 0

        debouncer.destroy()

    def test_ignores_messages_from_non_human_authors(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        event = create_message_event(
            conversation_id=CONV_A,
            author_id=AUTHOR_A,
            author_type="artificer",
            text="AI response",
        )

        debouncer.handle_event(event)
        timer.advance(5000)
        assert len(batches) == 0

        debouncer.destroy()


class TestDebouncerAdaptiveTiming:
    """Adaptive timing tests."""

    ADAPTIVE_CONFIG = DebouncerConfig(
        base_ms=3000,
        min_ms=1500,
        max_ms=8000,
        adaptive=True,
        history_window=20,
    )

    def test_adjusts_timer_based_on_message_cadence_after_sufficient_data(
        self,
    ) -> None:
        timer = FakeTimerProvider()
        d = Debouncer(self.ADAPTIVE_CONFIG, timer_provider=timer)
        b = collect_batches(d)

        # Send 4 messages at exactly 5000ms intervals.
        # msg0 at t=0 -> timer=3000 (0 intervals, uses base_ms)
        d.handle_event(make_message(text="0"))
        timer.advance(3000)  # seal at base_ms
        # t=3000, 0 intervals

        # Wait so next handle_event is at t=5000 -> interval=5000
        timer.advance(2000)
        d.handle_event(make_message(text="1"))
        timer.advance(3000)  # seal (1 interval, uses base_ms)
        # t=8000

        timer.advance(2000)
        d.handle_event(make_message(text="2"))
        timer.advance(3000)  # seal (2 intervals, uses base_ms)
        # t=13000

        timer.advance(2000)
        d.handle_event(make_message(text="3"))
        # 3 intervals: [5000, 5000, 5000]
        # Adaptive = median(5000) * 1.5 = 7500
        # Clamped: max(1500, min(8000, 7500)) = 7500
        timer.advance(7500)  # seal

        assert len(b) == 4

        # Now the 5th message should also use adaptive timing
        b.clear()
        timer.advance(2000)
        d.handle_event(make_message(text="4"))
        # 4 intervals: [5000, 5000, 5000, 9500]
        # Sorted: [5000, 5000, 5000, 9500], median = 5000
        # Adaptive = 5000*1.5 = 7500
        timer.advance(7499)
        assert len(b) == 0
        timer.advance(1)
        assert len(b) == 1

        d.destroy()

    def test_uses_base_ms_with_fewer_than_3_data_points(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(self.ADAPTIVE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        # First message -- no intervals yet -> base_ms
        debouncer.handle_event(make_message(text="first"))
        timer.advance(3000)
        assert len(batches) == 1

        # Second message -- 1 interval recorded -> base_ms
        debouncer.handle_event(make_message(text="second"))
        timer.advance(3000)
        assert len(batches) == 2

        # Third message -- 2 intervals recorded (still < 3) -> base_ms
        debouncer.handle_event(make_message(text="third"))
        timer.advance(3000)
        assert len(batches) == 3

        debouncer.destroy()

    def test_computes_median_correctly_with_outliers(self) -> None:
        timer = FakeTimerProvider()
        d = Debouncer(DebouncerConfig(base_ms=3000, min_ms=1500, max_ms=8000, adaptive=True), timer_provider=timer)
        b = collect_batches(d)

        # Build 4 messages with intervals of 4000ms each (3000 seal + 1000 gap).
        d.handle_event(make_message(text="0"))
        timer.advance(3000)  # seal
        timer.advance(1000)  # gap

        d.handle_event(make_message(text="1"))
        timer.advance(3000)
        timer.advance(1000)

        d.handle_event(make_message(text="2"))
        timer.advance(3000)
        timer.advance(1000)

        # msg3: 3 intervals [4000, 4000, 4000]. Adaptive=6000.
        d.handle_event(make_message(text="3"))
        timer.advance(6000)  # seal with adaptive

        # Now add an outlier gap of 20000ms
        timer.advance(20000)

        # msg4: 4 intervals [4000, 4000, 4000, 26000].
        # Sorted: [4000, 4000, 4000, 26000]. Median = (4000+4000)/2 = 4000.
        # Adaptive = 4000*1.5 = 6000. Clamped: 6000.
        d.handle_event(make_message(text="4"))
        timer.advance(6000)  # seal

        assert len(b) == 5

        # Verify: despite the outlier, median stays at 4000 -> adaptive = 6000
        b.clear()
        timer.advance(1000)
        d.handle_event(make_message(text="test"))
        timer.advance(5999)
        assert len(b) == 0
        timer.advance(1)
        assert len(b) == 1

        d.destroy()

    def test_clamps_to_min_ms(self) -> None:
        timer = FakeTimerProvider()
        d = Debouncer(
            DebouncerConfig(base_ms=100, min_ms=1500, max_ms=8000, adaptive=True),
            timer_provider=timer,
        )
        b = collect_batches(d)

        # Send messages with small intervals.
        d.handle_event(make_message(text="0"))
        timer.advance(100)  # seal (base_ms, 0 intervals)

        timer.advance(100)
        d.handle_event(make_message(text="1"))
        timer.advance(100)  # seal (base_ms, 1 interval of 200)

        timer.advance(100)
        d.handle_event(make_message(text="2"))
        timer.advance(100)  # seal (base_ms, 2 intervals)

        timer.advance(100)
        d.handle_event(make_message(text="3"))
        # 3 intervals: [200, 200, 200]. Adaptive = 300 -> clamped to 1500.
        timer.advance(1500)  # seal

        timer.advance(100)
        d.handle_event(make_message(text="4"))
        timer.advance(1500)  # seal

        assert len(b) == 5

        # Verify next message uses min_ms=1500
        b.clear()
        timer.advance(100)
        d.handle_event(make_message(text="final"))
        timer.advance(1499)
        assert len(b) == 0
        timer.advance(1)
        assert len(b) == 1

        d.destroy()

    def test_clamps_to_max_ms(self) -> None:
        timer = FakeTimerProvider()
        d = Debouncer(
            DebouncerConfig(base_ms=100, min_ms=1500, max_ms=8000, adaptive=True),
            timer_provider=timer,
        )
        b = collect_batches(d)

        # Send 4 messages to build 3 intervals, seal each
        d.handle_event(make_message(text="0"))
        timer.advance(100)  # seal (base_ms, 0 intervals)

        timer.advance(6900)  # gap -> interval = 7000
        d.handle_event(make_message(text="1"))
        timer.advance(100)  # seal (base_ms, 1 interval)

        timer.advance(6900)
        d.handle_event(make_message(text="2"))
        timer.advance(100)  # seal (base_ms, 2 intervals)

        timer.advance(6900)
        d.handle_event(make_message(text="3"))
        # Now 3 intervals of 7000. Adaptive = 10500 -> 8000.
        timer.advance(8000)  # seal

        assert len(b) == 4

        # Verify the next message uses max_ms
        b.clear()
        timer.advance(6900)
        d.handle_event(make_message(text="test"))
        timer.advance(7999)
        assert len(b) == 0
        timer.advance(1)
        assert len(b) == 1

        d.destroy()

    def test_always_uses_base_ms_when_adaptive_is_disabled(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(
            DebouncerConfig(
                base_ms=2000, min_ms=1500, max_ms=8000, adaptive=False, history_window=20
            ),
            timer_provider=timer,
        )
        batches = collect_batches(debouncer)

        # Build lots of history
        for i in range(10):
            debouncer.handle_event(make_message(text=str(i)))
            timer.advance(2000)

        batches.clear()

        # The 11th message should still use base_ms=2000
        debouncer.handle_event(make_message(text="final"))
        timer.advance(1999)
        assert len(batches) == 0
        timer.advance(1)
        assert len(batches) == 1

        debouncer.destroy()


class TestDebouncerDestroy:
    """destroy() behavior."""

    def test_clears_all_timers_on_destroy(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        debouncer.handle_event(make_message(author_id=AUTHOR_A, text="a"))
        debouncer.handle_event(make_message(author_id=AUTHOR_B, text="b"))

        debouncer.destroy()

        # Advancing time should not seal any batches
        timer.advance(10000)
        assert len(batches) == 0

    def test_ignores_events_after_destroy(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        debouncer.destroy()
        debouncer.handle_event(make_message(text="ignored"))

        timer.advance(10000)
        assert len(batches) == 0

    def test_ignores_typing_start_after_destroy(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        debouncer.destroy()

        # Should not throw
        debouncer.handle_typing_start(CONV_A, AUTHOR_A)


class TestDebouncerListeners:
    """Listener management."""

    def test_supports_removing_listeners_with_off(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches: list[SealedBatch] = []
        listener = lambda b: batches.append(b)

        debouncer.on("batch_sealed", listener)
        debouncer.handle_event(make_message(text="msg-1"))
        timer.advance(3000)
        assert len(batches) == 1

        debouncer.off("batch_sealed", listener)
        debouncer.handle_event(make_message(text="msg-2"))
        timer.advance(3000)
        assert len(batches) == 1  # Listener was removed

        debouncer.destroy()

    def test_emits_to_multiple_listeners(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        results1: list[SealedBatch] = []
        results2: list[SealedBatch] = []

        debouncer.on("batch_sealed", lambda b: results1.append(b))
        debouncer.on("batch_sealed", lambda b: results2.append(b))

        debouncer.handle_event(make_message(text="multi"))
        timer.advance(3000)

        assert len(results1) == 1
        assert len(results2) == 1

        debouncer.destroy()


class TestDebouncerBatchProperties:
    """Sealed batch properties."""

    def test_generates_a_unique_batch_id_for_each_sealed_batch(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        debouncer.handle_event(make_message(text="batch-1"))
        timer.advance(3000)

        debouncer.handle_event(make_message(text="batch-2"))
        timer.advance(3000)

        assert len(batches) == 2
        assert batches[0].batch_id
        assert batches[1].batch_id
        assert batches[0].batch_id != batches[1].batch_id

        debouncer.destroy()

    def test_sets_sealed_at_to_current_time_when_sealing(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        debouncer.handle_event(make_message(text="timed"))
        timer.advance(3000)

        assert len(batches) == 1
        assert isinstance(batches[0].sealed_at, (int, float))
        assert batches[0].sealed_at > 0

        debouncer.destroy()


class TestDebouncerInterleaving:
    """Interleaving: A's timer does not affect B's batch."""

    def test_does_not_let_author_a_affect_author_b_timing(self) -> None:
        timer = FakeTimerProvider()
        debouncer = Debouncer(BASE_CONFIG, timer_provider=timer)
        batches = collect_batches(debouncer)

        # t=0: A sends, A timer fires at t=3000
        msg_a = make_message(author_id=AUTHOR_A, text="A message")
        debouncer.handle_event(msg_a)

        # t=1500: B sends, B timer fires at t=4500
        timer.advance(1500)
        msg_b = make_message(author_id=AUTHOR_B, text="B message")
        debouncer.handle_event(msg_b)

        # t=2000: A sends again, A timer reset to fire at t=5000
        timer.advance(500)
        msg_a2 = make_message(author_id=AUTHOR_A, text="A again")
        debouncer.handle_event(msg_a2)

        # Advance to t=4500: B's timer fires (1500 + 3000)
        timer.advance(2500)
        assert len(batches) == 1
        assert batches[0].author_id == AUTHOR_B

        # Advance to t=5000: A's timer fires (2000 + 3000)
        timer.advance(500)
        assert len(batches) == 2
        assert batches[1].author_id == AUTHOR_A
        assert batches[1].event_ids == [msg_a.id, msg_a2.id]

        debouncer.destroy()
