"""Dispatch Queue tests -- Phase 6: Artificer Runtime (§11 of the IECP spec)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from iecp_core.artificer import DispatchQueue
from iecp_core.debounce.types import SealedBatch
from iecp_core.lock.types import LockState
from iecp_core.orchestrator.types import DispatchPayload
from iecp_core.types.entity import EntityId
from iecp_core.types.event import BatchId, ConversationId, EventId

_NOW = datetime.now(timezone.utc).isoformat()


def make_dispatch(id: str) -> DispatchPayload:
    conv_id = ConversationId(f"conv-{id}")
    entity_id = EntityId(f"entity-{id}")
    batch_id = BatchId(f"batch-{id}")
    return DispatchPayload(
        conversation_id=conv_id,
        entity_id=entity_id,
        batch=SealedBatch(
            batch_id=batch_id,
            conversation_id=conv_id,
            author_id=EntityId(f"author-{id}"),
            event_ids=[],
            sealed_at=1_000_000.0,
            message_count=1,
        ),
        lock=LockState(
            conversation_id=conv_id,
            holder_id=entity_id,
            acquired_at=1_000_000.0,
            ttl_ms=30_000,
            estimated_ms=20_000,
            expires_at=1_030_000.0,
            metadata={},
        ),
        ai_depth_counter=0,
        trace_id=f"trace-{id}",
    )


class TestDispatchQueue:
    async def test_processes_dispatches_up_to_max_concurrent(self) -> None:
        queue = DispatchQueue(2)
        order: list[str] = []

        def make_handler(id_: str, delay_ms: int):
            async def handler(dispatch: DispatchPayload) -> None:
                order.append(f"start-{id_}")
                await asyncio.sleep(delay_ms / 1000.0)
                order.append(f"end-{id_}")

            return handler

        p1 = asyncio.ensure_future(queue.enqueue(make_dispatch("1"), make_handler("1", 50)))
        p2 = asyncio.ensure_future(queue.enqueue(make_dispatch("2"), make_handler("2", 50)))
        p3 = asyncio.ensure_future(queue.enqueue(make_dispatch("3"), make_handler("3", 10)))

        # Yield control so tasks start
        await asyncio.sleep(0)

        stats = queue.get_stats()
        assert stats.active == 2
        assert stats.queued == 1
        assert stats.max_concurrent == 2

        await asyncio.gather(p1, p2, p3)

        assert "start-1" in order
        assert "end-1" in order
        assert "start-3" in order
        assert "end-3" in order

    async def test_queues_excess_dispatches(self) -> None:
        queue = DispatchQueue(1)
        block_event = asyncio.Event()
        started: list[str] = []

        async def blocking_handler(dispatch: DispatchPayload) -> None:
            await block_event.wait()

        p1 = asyncio.ensure_future(queue.enqueue(make_dispatch("1"), blocking_handler))

        async def second_handler(dispatch: DispatchPayload) -> None:
            started.append("2")

        p2 = asyncio.ensure_future(queue.enqueue(make_dispatch("2"), second_handler))

        # Yield to let tasks start
        await asyncio.sleep(0)

        assert queue.get_stats().active == 1
        assert queue.get_stats().queued == 1
        assert started == []

        # Unblock first
        block_event.set()
        await p1
        await p2

        assert "2" in started

    async def test_dequeues_when_active_dispatch_completes(self) -> None:
        queue = DispatchQueue(1)
        completed: list[str] = []

        async def handler_1(dispatch: DispatchPayload) -> None:
            completed.append("1")

        async def handler_2(dispatch: DispatchPayload) -> None:
            completed.append("2")

        p1 = asyncio.ensure_future(queue.enqueue(make_dispatch("1"), handler_1))
        p2 = asyncio.ensure_future(queue.enqueue(make_dispatch("2"), handler_2))

        await asyncio.gather(p1, p2)
        # Allow microtask for finally block to settle
        await asyncio.sleep(0)

        assert completed == ["1", "2"]
        assert queue.get_stats().active == 0
        assert queue.get_stats().queued == 0

    def test_get_stats_returns_correct_counts(self) -> None:
        queue = DispatchQueue(3)
        stats = queue.get_stats()
        assert stats.active == 0
        assert stats.queued == 0
        assert stats.max_concurrent == 3
