"""Debouncer Types -- Phase 2 of the IECP protocol.

Defines configuration, sealed batch structure, and timer abstraction
for the smart debouncing engine.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import BaseModel

from ..types.entity import EntityId
from ..types.event import BatchId, ConversationId, EventId


# -- Configuration -----------------------------------------------------------


@dataclass(frozen=True)
class DebouncerConfig:
    """Configuration for the debounce engine. All timings in milliseconds."""

    base_ms: int = 3000
    """Base debounce delay."""

    min_ms: int = 1500
    """Minimum adaptive delay."""

    max_ms: int = 8000
    """Maximum adaptive delay."""

    adaptive: bool = True
    """Enable adaptive timing based on author cadence."""

    history_window: int = 20
    """Number of inter-message intervals to track for adaptation."""


DEFAULT_DEBOUNCER_CONFIG = DebouncerConfig()


# -- Sealed Batch ------------------------------------------------------------


class SealedBatch(BaseModel):
    """A sealed batch -- a group of messages from a single author
    that have been debounced and are ready for AI dispatch."""

    batch_id: BatchId
    conversation_id: ConversationId
    author_id: EntityId
    event_ids: list[EventId]
    sealed_at: float
    """Unix timestamp (ms) when the batch was sealed."""
    message_count: int


# -- Timer Abstraction -------------------------------------------------------


@runtime_checkable
class TimerProvider(Protocol):
    """Timer interface -- abstracts setTimeout/clearTimeout so the
    implementation can be swapped for Redis-backed timers later."""

    def set_timeout(self, callback: Callable[[], None], ms: float) -> Any:
        """Schedule *callback* to fire after *ms* milliseconds. Returns a handle."""
        ...

    def clear_timeout(self, handle: Any) -> None:
        """Cancel a previously scheduled timeout."""
        ...

    def now(self) -> float:
        """Return the current time in milliseconds."""
        ...


class DefaultTimerProvider:
    """Default timer provider using threading.Timer and time.time."""

    def set_timeout(self, callback: Callable[[], None], ms: float) -> threading.Timer:
        t = threading.Timer(ms / 1000.0, callback)
        t.daemon = True
        t.start()
        return t

    def clear_timeout(self, handle: Any) -> None:
        if isinstance(handle, threading.Timer):
            handle.cancel()

    def now(self) -> float:
        return time.time() * 1000.0


default_timer_provider = DefaultTimerProvider()
