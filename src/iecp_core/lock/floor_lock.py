"""FloorLock -- Mutual exclusion for AI responses in IECP conversations.

Implements the Floor Lock mechanism: exactly one AI entity holds the lock
per conversation at any time. Uses a TimerProvider for testable timers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..debounce.types import TimerProvider, default_timer_provider
from ..types.entity import EntityId
from ..types.event import ConversationId
from .types import (
    DEFAULT_FLOOR_LOCK_CONFIG,
    PRIORITY_ORDER,
    FloorLockConfig,
    LockAcquiredEvent,
    LockRelease,
    LockReleaseReason,
    LockReleasedEvent,
    LockRequest,
    LockResult,
    LockState,
    QueueEntry,
)

# -- Internal State ----------------------------------------------------------


@dataclass
class _ConversationLockState:
    """Internal per-conversation lock state."""

    lock: LockState | None = None
    queue: list[QueueEntry] = field(default_factory=list)
    last_served: EntityId | None = None
    ttl_timer: Any = None
    grace_timer: Any = None
    in_grace_period: bool = False


# -- FloorLock ---------------------------------------------------------------


class FloorLock:
    """The Floor Lock enforces mutual exclusion: exactly one AI entity
    holds the lock per conversation at any time.

    Usage::

        lock = FloorLock(config)
        lock.on('lock_acquired', lambda state: ...)
        lock.on('lock_released', lambda release: ...)
        result = lock.acquire(request)
        lock.release(conversation_id, entity_id, 'commit')
        lock.destroy()
    """

    def __init__(
        self,
        config: FloorLockConfig | dict[str, Any] | None = None,
        timer_provider: TimerProvider | None = None,
    ) -> None:
        if config is None:
            self._config = DEFAULT_FLOOR_LOCK_CONFIG
        elif isinstance(config, dict):
            self._config = FloorLockConfig(**config)
        else:
            self._config = config

        self._timer: TimerProvider = timer_provider or default_timer_provider

        self._conversations: dict[ConversationId, _ConversationLockState] = {}
        self._listeners: dict[str, list[Callable[..., Any]]] = {
            "lock_acquired": [],
            "lock_released": [],
        }

    # -- Event Emitter -------------------------------------------------------

    def on(self, event: str, listener: Callable[..., Any]) -> None:
        """Register a listener for lock events."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def off(self, event: str, listener: Callable[..., Any]) -> None:
        """Remove a listener."""
        listeners = self._listeners.get(event)
        if listeners and listener in listeners:
            listeners.remove(listener)

    def _emit(self, event: str, payload: Any) -> None:
        """Emit an event to all registered listeners."""
        listeners = self._listeners.get(event)
        if listeners:
            for listener in listeners:
                listener(payload)

    # -- State Helpers -------------------------------------------------------

    def _get_or_create(self, conversation_id: ConversationId) -> _ConversationLockState:
        state = self._conversations.get(conversation_id)
        if state is None:
            state = _ConversationLockState()
            self._conversations[conversation_id] = state
        return state

    def _compute_ttl(self, estimated_ms: float) -> float:
        """Compute actual TTL: min(max(estimated * multiplier, default), max)."""
        computed = estimated_ms * self._config.ttl_multiplier
        floored = max(computed, self._config.default_ttl_ms)
        return min(floored, self._config.max_ttl_ms)

    # -- Timer Management ----------------------------------------------------

    def _clear_timers(self, state: _ConversationLockState) -> None:
        if state.ttl_timer is not None:
            self._timer.clear_timeout(state.ttl_timer)
            state.ttl_timer = None
        if state.grace_timer is not None:
            self._timer.clear_timeout(state.grace_timer)
            state.grace_timer = None
        state.in_grace_period = False

    def _start_ttl_timer(
        self, conversation_id: ConversationId, state: _ConversationLockState
    ) -> None:
        if state.lock is None:
            return

        ttl_ms = state.lock.ttl_ms

        def on_ttl_expired() -> None:
            state.ttl_timer = None
            state.in_grace_period = True

            def on_grace_expired() -> None:
                state.grace_timer = None
                state.in_grace_period = False
                if state.lock is not None:
                    self._release_lock_internal(conversation_id, state, "ttl_expired")

            state.grace_timer = self._timer.set_timeout(
                on_grace_expired, self._config.grace_period_ms
            )

        state.ttl_timer = self._timer.set_timeout(on_ttl_expired, ttl_ms)

    # -- Acquire -------------------------------------------------------------

    def acquire(self, request: LockRequest) -> LockResult:
        """Request the Floor Lock for a conversation.

        - Free lock -> grant immediately.
        - Held by same entity -> extend TTL.
        - Held by another -> deny with queue position.
        """
        state = self._get_or_create(request.conversation_id)
        now = self._timer.now()

        # Purge expired queue entries
        self._purge_expired_queue_entries(state, now)

        # Case 1: Lock is free
        if state.lock is None:
            return self._grant_lock(request, state, now)

        # Case 2: Same entity re-acquires -> extend
        if state.lock.holder_id == request.entity_id:
            return self._extend_lock(request, state, now)

        # Case 3: Held by another -> deny, queue
        return self._deny_and_queue(request, state, now)

    def _grant_lock(
        self,
        request: LockRequest,
        state: _ConversationLockState,
        now: float,
    ) -> LockResult:
        ttl_ms = self._compute_ttl(request.estimated_ms)
        lock = LockState(
            conversation_id=request.conversation_id,
            holder_id=request.entity_id,
            acquired_at=now,
            ttl_ms=ttl_ms,
            estimated_ms=request.estimated_ms,
            expires_at=now + ttl_ms,
            metadata=request.metadata or {},
        )

        state.lock = lock
        state.last_served = request.entity_id

        # Remove from queue if present
        self._remove_from_queue(state, request.entity_id)

        self._start_ttl_timer(request.conversation_id, state)
        self._emit("lock_acquired", lock)

        return LockResult(granted=True, lock=lock)

    def _extend_lock(
        self,
        request: LockRequest,
        state: _ConversationLockState,
        now: float,
    ) -> LockResult:
        self._clear_timers(state)

        old_lock = state.lock
        assert old_lock is not None

        ttl_ms = self._compute_ttl(request.estimated_ms)
        merged_metadata = {**old_lock.metadata, **(request.metadata or {})}

        lock = LockState(
            conversation_id=old_lock.conversation_id,
            holder_id=old_lock.holder_id,
            acquired_at=old_lock.acquired_at,
            ttl_ms=ttl_ms,
            estimated_ms=request.estimated_ms,
            expires_at=now + ttl_ms,
            metadata=merged_metadata,
        )

        state.lock = lock
        self._start_ttl_timer(request.conversation_id, state)

        return LockResult(granted=True, lock=lock)

    def _deny_and_queue(
        self,
        request: LockRequest,
        state: _ConversationLockState,
        now: float,
    ) -> LockResult:
        entry = QueueEntry(
            entity_id=request.entity_id,
            priority=request.priority,
            queued_at=now,
            estimated_ms=request.estimated_ms,
            metadata=request.metadata or {},
            expires_at=now + self._config.max_ttl_ms,
        )

        # Check if already in queue -- update if so
        existing_idx = None
        for i, e in enumerate(state.queue):
            if e.entity_id == request.entity_id:
                existing_idx = i
                break

        if existing_idx is not None:
            state.queue[existing_idx] = entry
        else:
            state.queue.append(entry)

        # Sort queue
        self._sort_queue(state)

        position = next(
            i for i, e in enumerate(state.queue) if e.entity_id == request.entity_id
        )

        return LockResult(
            granted=False,
            reason="lock_held",
            queue_position=position + 1,
        )

    def _sort_queue(self, state: _ConversationLockState) -> None:
        """Sort queue by priority (desc), round-robin, then FIFO."""
        last_served = state.last_served

        def sort_key(entry: QueueEntry) -> tuple[int, int, float]:
            pri = -PRIORITY_ORDER[entry.priority]
            # Round-robin: last served goes to end within same priority
            rr = 1 if (last_served is not None and entry.entity_id == last_served) else 0
            return (pri, rr, entry.queued_at)

        state.queue.sort(key=sort_key)

    def _remove_from_queue(self, state: _ConversationLockState, entity_id: EntityId) -> None:
        state.queue = [e for e in state.queue if e.entity_id != entity_id]

    def _purge_expired_queue_entries(
        self, state: _ConversationLockState, now: float
    ) -> None:
        state.queue = [e for e in state.queue if e.expires_at > now]

    # -- Release -------------------------------------------------------------

    def release(
        self,
        conversation_id: ConversationId,
        entity_id: EntityId,
        reason: LockReleaseReason,
    ) -> bool:
        """Release the Floor Lock for a conversation.

        Only the current holder can release. Returns True if released.
        """
        state = self._conversations.get(conversation_id)
        if state is None or state.lock is None:
            return False
        if state.lock.holder_id != entity_id:
            return False

        self._release_lock_internal(conversation_id, state, reason)
        return True

    def _release_lock_internal(
        self,
        conversation_id: ConversationId,
        state: _ConversationLockState,
        reason: LockReleaseReason,
    ) -> None:
        lock = state.lock
        assert lock is not None

        self._clear_timers(state)
        state.lock = None

        release = LockRelease(
            conversation_id=conversation_id,
            entity_id=lock.holder_id,
            reason=reason,
        )

        self._emit(
            "lock_released",
            LockReleasedEvent(
                conversation_id=release.conversation_id,
                entity_id=release.entity_id,
                reason=release.reason,
                state=lock,
            ),
        )

        # Auto-grant to next in queue
        self._grant_next_in_queue(conversation_id, state)

    def _grant_next_in_queue(
        self,
        conversation_id: ConversationId,
        state: _ConversationLockState,
    ) -> None:
        now = self._timer.now()
        self._purge_expired_queue_entries(state, now)
        self._sort_queue(state)

        if len(state.queue) == 0:
            return

        next_entry = state.queue.pop(0)
        request = LockRequest(
            entity_id=next_entry.entity_id,
            conversation_id=conversation_id,
            estimated_ms=next_entry.estimated_ms,
            priority=next_entry.priority,
            metadata=next_entry.metadata,
        )

        self._grant_lock(request, state, now)

    # -- Human Interruption --------------------------------------------------

    def handle_human_interrupt(self, conversation_id: ConversationId) -> bool:
        """Handle a human interruption on a conversation.

        Immediately releases the lock, clears the queue, and emits
        lock_released with reason 'human_interrupt'.
        """
        state = self._conversations.get(conversation_id)
        if state is None or state.lock is None:
            return False

        self._clear_timers(state)

        lock = state.lock
        state.lock = None
        state.queue = []  # Clear queue -- human takes priority

        release = LockRelease(
            conversation_id=conversation_id,
            entity_id=lock.holder_id,
            reason="human_interrupt",
        )

        self._emit(
            "lock_released",
            LockReleasedEvent(
                conversation_id=release.conversation_id,
                entity_id=release.entity_id,
                reason=release.reason,
                state=lock,
            ),
        )

        return True

    # -- Queue Management ----------------------------------------------------

    def cancel_queue(
        self, conversation_id: ConversationId, entity_id: EntityId
    ) -> bool:
        """Cancel an entity's position in the wait queue. Returns True if found."""
        state = self._conversations.get(conversation_id)
        if state is None:
            return False

        before = len(state.queue)
        self._remove_from_queue(state, entity_id)
        return len(state.queue) < before

    # -- State Queries -------------------------------------------------------

    def get_lock_state(self, conversation_id: ConversationId) -> LockState | None:
        """Get the current lock state for a conversation."""
        state = self._conversations.get(conversation_id)
        if state is None:
            return None
        return state.lock

    def is_locked(self, conversation_id: ConversationId) -> bool:
        """Check if a conversation is currently locked."""
        state = self._conversations.get(conversation_id)
        return state is not None and state.lock is not None

    def get_queue_position(
        self, conversation_id: ConversationId, entity_id: EntityId
    ) -> int | None:
        """Get an entity's position in the wait queue (1-indexed)."""
        state = self._conversations.get(conversation_id)
        if state is None:
            return None

        for i, e in enumerate(state.queue):
            if e.entity_id == entity_id:
                return i + 1
        return None

    def get_queue_length(self, conversation_id: ConversationId) -> int:
        """Get the queue length for a conversation."""
        state = self._conversations.get(conversation_id)
        if state is None:
            return 0
        return len(state.queue)

    # -- Cleanup -------------------------------------------------------------

    def destroy(self) -> None:
        """Destroy the FloorLock, clearing all timers and state."""
        for state in self._conversations.values():
            self._clear_timers(state)
        self._conversations.clear()
        self._listeners["lock_acquired"] = []
        self._listeners["lock_released"] = []
