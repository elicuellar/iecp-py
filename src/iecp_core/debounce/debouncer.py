"""Debouncer -- Smart debouncing engine for IECP.

Groups rapid-fire human message fragments into sealed batches
before AI dispatch. Each author+conversation pair has an
independent timer and adaptive cadence tracker.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from ..types.entity import EntityId
from ..types.event import BatchId, ConversationId, Event, EventId
from ..utils import generate_id
from .types import (
    DEFAULT_DEBOUNCER_CONFIG,
    DebouncerConfig,
    SealedBatch,
    TimerProvider,
    default_timer_provider,
)

# -- Internal Types ----------------------------------------------------------

# SlotKey is just a string: "{conversation_id}::{author_id}"
SlotKey = str


@dataclass
class _ActiveBatch:
    """Active (unsealed) batch state."""

    conversation_id: ConversationId
    author_id: EntityId
    event_ids: list[EventId]
    timer_handle: Any
    last_event_is_continuation: bool
    arrival_timestamps: list[float] = field(default_factory=list)


@dataclass
class _CadenceHistory:
    """Per-author+conversation cadence history."""

    intervals: list[float] = field(default_factory=list)
    last_message_at: float | None = None


# -- Helpers -----------------------------------------------------------------


def _make_slot_key(conversation_id: ConversationId, author_id: EntityId) -> SlotKey:
    return f"{conversation_id}::{author_id}"


def _median(sorted_arr: list[float]) -> float:
    """Compute the median of a sorted numeric list."""
    n = len(sorted_arr)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return sorted_arr[mid]
    return (sorted_arr[mid - 1] + sorted_arr[mid]) / 2.0


# -- Debouncer ---------------------------------------------------------------


class Debouncer:
    """Smart debouncing engine for IECP.

    Groups rapid-fire human message fragments into sealed batches
    before AI dispatch. Each author+conversation pair has an
    independent timer and adaptive cadence tracker.
    """

    def __init__(
        self,
        config: DebouncerConfig | dict[str, Any] | None = None,
        timer_provider: TimerProvider | None = None,
    ) -> None:
        if config is None:
            self._config = DEFAULT_DEBOUNCER_CONFIG
        elif isinstance(config, dict):
            self._config = DebouncerConfig(**config)
        else:
            self._config = config

        self._timer: TimerProvider = timer_provider or default_timer_provider

        self._active_batches: dict[SlotKey, _ActiveBatch] = {}
        self._cadence_history: dict[SlotKey, _CadenceHistory] = {}
        self._listeners: dict[str, set[Callable[..., Any]]] = {}
        self._destroyed: bool = False

        # Capture the event loop at construction time for cross-thread _emit.
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    # -- Public API ----------------------------------------------------------

    def on(self, event: str, listener: Callable[..., Any]) -> None:
        """Register a listener for a debouncer event."""
        if event not in self._listeners:
            self._listeners[event] = set()
        self._listeners[event].add(listener)

    def off(self, event: str, listener: Callable[..., Any]) -> None:
        """Remove a listener for a debouncer event."""
        listeners = self._listeners.get(event)
        if listeners:
            listeners.discard(listener)

    async def handle_event(self, event: Event) -> None:
        """Handle an incoming event from the event log.

        Only message events from human authors are debounced.
        All other events are ignored.
        """
        if self._destroyed:
            return

        # Only debounce human message events
        if event.type != "message":
            return
        if event.author_type != "human":
            return

        key = _make_slot_key(event.conversation_id, event.author_id)
        now = self._timer.now()

        # Update cadence history
        self._update_cadence_history(key, now)

        existing = self._active_batches.get(key)

        if existing is not None:
            # Add to existing batch, reset timer
            existing.event_ids.append(event.id)
            existing.last_event_is_continuation = event.is_continuation
            self._timer.clear_timeout(existing.timer_handle)
            existing.timer_handle = self._start_timer(key)
        else:
            # Create new batch
            batch = _ActiveBatch(
                conversation_id=event.conversation_id,
                author_id=event.author_id,
                event_ids=[event.id],
                timer_handle=None,
                last_event_is_continuation=event.is_continuation,
            )
            batch.timer_handle = self._start_timer(key)
            self._active_batches[key] = batch

    async def handle_typing_start(
        self, conversation_id: ConversationId, author_id: EntityId
    ) -> None:
        """Handle a typing_start signal from a human.

        Resets the debounce timer if an active batch exists.
        Does nothing if no active batch.
        """
        if self._destroyed:
            return

        key = _make_slot_key(conversation_id, author_id)
        existing = self._active_batches.get(key)
        if existing is None:
            return

        # Reset the timer
        self._timer.clear_timeout(existing.timer_handle)
        existing.timer_handle = self._start_timer(key)

    async def destroy(self) -> None:
        """Destroy the debouncer -- clear all active timers and state.

        After calling destroy(), the debouncer will not process
        any further events.
        """
        self._destroyed = True
        for batch in self._active_batches.values():
            self._timer.clear_timeout(batch.timer_handle)
        self._active_batches.clear()
        self._cadence_history.clear()
        self._listeners.clear()

    # -- Internal ------------------------------------------------------------

    def _compute_delay(self, key: SlotKey) -> float:
        """Compute the adaptive debounce delay for a given slot."""
        if not self._config.adaptive:
            return float(self._config.base_ms)

        history = self._cadence_history.get(key)
        if not history or len(history.intervals) < 3:
            return float(self._config.base_ms)

        # Compute median of recorded intervals
        sorted_intervals = sorted(history.intervals)
        med = _median(sorted_intervals)

        # New timer = median * 1.5, clamped to [min_ms, max_ms]
        adaptive = round(med * 1.5)
        return float(max(self._config.min_ms, min(self._config.max_ms, adaptive)))

    def _start_timer(self, key: SlotKey) -> Any:
        """Start a debounce timer for a given slot. Returns the timer handle."""
        delay = self._compute_delay(key)
        return self._timer.set_timeout(lambda: self._on_timer_fire(key), delay)

    def _on_timer_fire(self, key: SlotKey) -> None:
        """Called when a debounce timer fires.

        This is called synchronously by the timer provider. It handles
        the batch sealing logic synchronously, but uses _emit_sync to
        dispatch to listeners (which may schedule async work).
        """
        if self._destroyed:
            return

        batch = self._active_batches.get(key)
        if batch is None:
            return

        # If the last message has is_continuation, extend the timer
        if batch.last_event_is_continuation:
            batch.timer_handle = self._start_timer(key)
            return

        # Seal the batch
        self._seal_batch(key, batch)

    def _seal_batch(self, key: SlotKey, batch: _ActiveBatch) -> None:
        """Seal an active batch and emit the batch_sealed event."""
        del self._active_batches[key]

        sealed = SealedBatch(
            batch_id=BatchId(generate_id()),
            conversation_id=batch.conversation_id,
            author_id=batch.author_id,
            event_ids=list(batch.event_ids),
            sealed_at=self._timer.now(),
            message_count=len(batch.event_ids),
        )

        self._emit("batch_sealed", sealed)

    def _update_cadence_history(self, key: SlotKey, now: float) -> None:
        """Update cadence history for adaptive timing."""
        history = self._cadence_history.get(key)
        if history is None:
            history = _CadenceHistory()
            self._cadence_history[key] = history

        if history.last_message_at is not None:
            interval = now - history.last_message_at
            history.intervals.append(interval)

            # Cap at history_window
            if len(history.intervals) > self._config.history_window:
                history.intervals.pop(0)

        history.last_message_at = now

    def _emit(self, event: str, *args: Any) -> None:
        """Emit a debouncer event to all registered listeners.

        Supports both sync and async listeners. Async listeners
        are scheduled as tasks on the running event loop, or via
        ``call_soon_threadsafe`` when called from a timer thread.
        """
        listeners = self._listeners.get(event)
        if not listeners:
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        for listener in list(listeners):
            if asyncio.iscoroutinefunction(listener):
                if running_loop is not None:
                    asyncio.ensure_future(listener(*args))
                elif self._loop is not None and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(listener(*args), self._loop)
            else:
                listener(*args)
