"""Dispatch Queue -- §11 of the IECP specification.

Manages concurrent invocation limits for artificer generation.
Queues excess dispatches and processes them as slots open up.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..orchestrator.types import DispatchPayload


# -- Types --------------------------------------------------------------------


@dataclass
class DispatchQueueStats:
    """Current queue statistics."""

    active: int
    queued: int
    max_concurrent: int


# -- DispatchQueue ------------------------------------------------------------


class DispatchQueue:
    """Manages concurrent invocation limits for artificer generation."""

    def __init__(self, max_concurrent: int) -> None:
        self._max_concurrent = max_concurrent
        self._active = 0
        self._queue: asyncio.Queue[
            tuple[
                DispatchPayload,
                Callable[[DispatchPayload], Awaitable[None]],
                asyncio.Future[None],
            ]
        ] = asyncio.Queue()
        self._queued_count = 0

    async def enqueue(
        self,
        dispatch: DispatchPayload,
        handler: Callable[[DispatchPayload], Awaitable[None]],
    ) -> None:
        """Enqueue a dispatch.

        Returns a coroutine that resolves when the dispatch is processed
        (either completed or raises on failure).
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[None] = loop.create_future()

        self._queue.put_nowait((dispatch, handler, future))
        self._queued_count += 1

        self._process_next()

        await future

    def get_stats(self) -> DispatchQueueStats:
        """Get current queue stats."""
        return DispatchQueueStats(
            active=self._active,
            queued=self._queued_count,
            max_concurrent=self._max_concurrent,
        )

    # -- Private --------------------------------------------------------------

    def _process_next(self) -> None:
        """Try to start the next queued item if capacity allows."""
        if self._active >= self._max_concurrent or self._queue.empty():
            return

        try:
            dispatch, handler, future = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        self._queued_count -= 1
        self._active += 1

        task = asyncio.ensure_future(handler(dispatch))

        def _on_done(t: asyncio.Task[None]) -> None:
            self._active -= 1
            if not future.done():
                exc = t.exception()
                if exc is not None:
                    future.set_exception(exc)
                else:
                    future.set_result(None)
            self._process_next()

        task.add_done_callback(_on_done)
