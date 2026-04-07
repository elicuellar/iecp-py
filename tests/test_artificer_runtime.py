"""Artificer Runtime tests -- Phase 6: Artificer Runtime (§11 of the IECP spec)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from iecp_core.artificer import (
    ArtificerModelConfig,
    ArtificerPersona,
    ArtificerRuntime,
    ArtificerRuntimeConfig,
    MessageCommittedEvent,
    ModelMessage,
    ModelProvider,
    OutputFilter,
    RuntimeErrorEvent,
    StreamChunk,
    StreamChunkEvent,
)
from iecp_core.context.types import ContextPayload, ParticipantSummary
from iecp_core.debounce.types import SealedBatch
from iecp_core.lock.types import LockState
from iecp_core.orchestrator.types import DispatchPayload
from iecp_core.types.entity import EntityId
from iecp_core.types.event import (
    BatchId,
    ConversationId,
    Event,
    EventId,
    MessageContent,
)

_NOW = datetime.now(timezone.utc).isoformat()

CONV_ID = ConversationId("conv-1")
ENTITY_ID = EntityId("entity-meina")

PERSONA = ArtificerPersona(
    name="Meina",
    role="analyst",
    phase="Discovery",
    system_prompt="You are Meina, a Discovery analyst.",
)

MODEL_CONFIG = ArtificerModelConfig(
    base_url="https://api.example.com",
    api_key="test-key",
    model="test-model",
)


# -- Helpers ------------------------------------------------------------------


def make_dispatch() -> DispatchPayload:
    return DispatchPayload(
        conversation_id=CONV_ID,
        entity_id=ENTITY_ID,
        batch=SealedBatch(
            batch_id=BatchId("batch-1"),
            conversation_id=CONV_ID,
            author_id=EntityId("entity-alice"),
            event_ids=[EventId("evt-1")],
            sealed_at=1_000_000.0,
            message_count=1,
        ),
        lock=LockState(
            conversation_id=CONV_ID,
            holder_id=ENTITY_ID,
            acquired_at=1_000_000.0,
            ttl_ms=30_000,
            estimated_ms=20_000,
            expires_at=1_030_000.0,
            metadata={},
        ),
        ai_depth_counter=0,
        trace_id="trace-1",
    )


def make_context_payload() -> ContextPayload:
    return ContextPayload(
        conversation_id=CONV_ID,
        recipient_id=ENTITY_ID,
        unread_messages=[
            Event(
                id=EventId("evt-1"),
                conversation_id=CONV_ID,
                author_id=EntityId("entity-alice"),
                author_type="human",
                type="message",
                content=MessageContent(text="Hello Meina!", format="plain", mentions=[]),
                is_continuation=False,
                is_complete=True,
                ai_depth_counter=0,
                status="active",
                created_at=_NOW,
                metadata={},
            )
        ],
        recent_history=[],
        conversation_summary=None,
        participants=[
            ParticipantSummary(
                entity_id=ENTITY_ID,
                display_name="Meina",
                entity_type="artificer",
                capabilities=None,
                lifecycle_status="active",
            ),
            ParticipantSummary(
                entity_id=EntityId("entity-alice"),
                display_name="Alice",
                entity_type="human",
                capabilities=None,
                lifecycle_status="active",
            ),
        ],
        response_expected=True,
        batch_id=BatchId("batch-1"),
        your_role="Analyst",
        your_capabilities=["analysis"],
        your_instructions=None,
        active_decisions=[],
        pending_handoffs=[],
        token_budget=100_000,
        tokens_used=5_000,
    )


def make_mock_model_provider(chunks: list[str]) -> ModelProvider:
    """Create a mock model provider that yields the given text chunks."""

    class _MockProvider:
        def __init__(self) -> None:
            self.abort = MagicMock()

        async def stream(
            self,
            messages: list[ModelMessage],
            config: ArtificerModelConfig,
        ) -> AsyncIterator[StreamChunk]:
            for text in chunks:
                yield StreamChunk(text=text, done=False)
            yield StreamChunk(text="", done=True)

    return _MockProvider()  # type: ignore[return-value]


def make_failing_model_provider(error: Exception, fail_count: int) -> ModelProvider:
    """Create a model provider that fails the first `fail_count` calls."""

    class _FailingProvider:
        def __init__(self) -> None:
            self._calls = 0
            self.abort = MagicMock()

        async def stream(
            self,
            messages: list[ModelMessage],
            config: ArtificerModelConfig,
        ) -> AsyncIterator[StreamChunk]:
            self._calls += 1
            if self._calls <= fail_count:
                raise error
            yield StreamChunk(text="recovered", done=False)
            yield StreamChunk(text="", done=True)

    return _FailingProvider()  # type: ignore[return-value]


def make_mock_context_builder() -> Any:
    mock = MagicMock()
    mock.build_context = AsyncMock(return_value=make_context_payload())
    return mock


def make_mock_floor_lock() -> Any:
    mock = MagicMock()
    mock.release = AsyncMock(return_value=True)
    return mock


def make_mock_event_store() -> Any:
    mock = MagicMock()

    async def append_event(event: Event) -> Event:
        return event

    mock.append_event = AsyncMock(side_effect=append_event)
    return mock


def create_runtime(
    model_provider: ModelProvider,
    **config_overrides: Any,
) -> tuple[ArtificerRuntime, Any, Any, Any]:
    context_builder = make_mock_context_builder()
    floor_lock = make_mock_floor_lock()
    event_store = make_mock_event_store()

    defaults = dict(
        max_retries=3,
        retry_base_delay_ms=1,  # 1ms for fast tests
        stream_flush_interval_ms=10,
        max_concurrent_invocations=5,
    )
    defaults.update(config_overrides)
    config = ArtificerRuntimeConfig(**defaults)

    runtime = ArtificerRuntime(
        model_provider=model_provider,
        context_builder=context_builder,
        output_filter=OutputFilter(),
        floor_lock=floor_lock,
        event_store=event_store,
        config=config,
    )
    return runtime, context_builder, floor_lock, event_store


# -- Tests --------------------------------------------------------------------


class TestArtificerRuntime:
    async def test_registers_artificer_and_handles_dispatch(self) -> None:
        provider = make_mock_model_provider(["Hello!"])
        runtime, _, floor_lock, event_store = create_runtime(provider)
        runtime.register_artificer(ENTITY_ID, PERSONA, MODEL_CONFIG)

        await runtime.handle_dispatch(make_dispatch())

        assert event_store.append_event.called
        floor_lock.release.assert_called_with(CONV_ID, ENTITY_ID, "commit")

    async def test_builds_context_from_dispatch_payload(self) -> None:
        provider = make_mock_model_provider(["ok"])
        runtime, context_builder, _, _ = create_runtime(provider)
        runtime.register_artificer(ENTITY_ID, PERSONA, MODEL_CONFIG)

        await runtime.handle_dispatch(make_dispatch())

        context_builder.build_context.assert_called_once()
        call_args = context_builder.build_context.call_args[0][0]
        assert call_args.dispatch.conversation_id == CONV_ID
        assert call_args.system_prompt == PERSONA.system_prompt

    async def test_streams_chunks_and_emits_stream_chunk_events(self) -> None:
        provider = make_mock_model_provider(["Hello", " world"])
        runtime, _, _, _ = create_runtime(provider)
        runtime.register_artificer(ENTITY_ID, PERSONA, MODEL_CONFIG)

        chunk_events: list[StreamChunkEvent] = []
        runtime.on("stream_chunk", lambda e: chunk_events.append(e))

        await runtime.handle_dispatch(make_dispatch())

        assert len(chunk_events) == 2
        assert chunk_events[0].text == "Hello"
        assert chunk_events[1].text == " world"
        assert chunk_events[0].conversation_id == CONV_ID
        assert chunk_events[0].entity_id == ENTITY_ID

    async def test_commits_message_on_completion(self) -> None:
        provider = make_mock_model_provider(["Done"])
        runtime, _, _, event_store = create_runtime(provider)
        runtime.register_artificer(ENTITY_ID, PERSONA, MODEL_CONFIG)

        committed: list[MessageCommittedEvent] = []
        runtime.on("message_committed", lambda e: committed.append(e))

        await runtime.handle_dispatch(make_dispatch())

        assert len(committed) == 1
        assert committed[0].conversation_id == CONV_ID
        assert committed[0].entity_id == ENTITY_ID
        assert committed[0].event.type == "message"

    async def test_runs_output_filter_before_commit(self) -> None:
        provider = make_mock_model_provider(["Valid response"])
        runtime, _, _, event_store = create_runtime(provider)
        runtime.register_artificer(ENTITY_ID, PERSONA, MODEL_CONFIG)

        await runtime.handle_dispatch(make_dispatch())

        # Should have committed a message (filter passed)
        assert event_store.append_event.call_count == 1
        committed_event = event_store.append_event.call_args[0][0]
        assert committed_event.type == "message"

    async def test_rejects_output_that_fails_filter_posts_system_event(self) -> None:
        # Empty output will fail the filter
        provider = make_mock_model_provider([""])
        runtime, _, _, event_store = create_runtime(provider)
        runtime.register_artificer(ENTITY_ID, PERSONA, MODEL_CONFIG)

        await runtime.handle_dispatch(make_dispatch())

        # Should have committed a system event (filter rejection), not a message
        assert event_store.append_event.call_count == 1
        committed_event = event_store.append_event.call_args[0][0]
        assert committed_event.type == "system"

    async def test_retries_on_model_error_with_backoff(self) -> None:
        provider = make_failing_model_provider(Exception("Rate limit"), 2)
        runtime, _, _, event_store = create_runtime(provider)
        runtime.register_artificer(ENTITY_ID, PERSONA, MODEL_CONFIG)

        errors: list[RuntimeErrorEvent] = []
        runtime.on("error", lambda e: errors.append(e))

        await runtime.handle_dispatch(make_dispatch())

        # Should have had 2 errors, then succeeded on 3rd attempt
        assert len(errors) == 2
        assert errors[0].retry_count == 1
        assert errors[1].retry_count == 2

        # Should have committed a message (recovered)
        committed_event = event_store.append_event.call_args[0][0]
        assert committed_event.type == "message"

    async def test_posts_system_event_after_max_retries_exhausted(self) -> None:
        provider = make_failing_model_provider(Exception("Permanent failure"), 10)
        runtime, _, _, event_store = create_runtime(provider, max_retries=2)
        runtime.register_artificer(ENTITY_ID, PERSONA, MODEL_CONFIG)

        errors: list[RuntimeErrorEvent] = []
        runtime.on("error", lambda e: errors.append(e))

        await runtime.handle_dispatch(make_dispatch())

        # 3 errors (initial + 2 retries)
        assert len(errors) == 3

        # Should have committed a system event about the failure
        assert event_store.append_event.call_count == 1
        committed_event = event_store.append_event.call_args[0][0]
        assert committed_event.type == "system"
        assert isinstance(committed_event.content.description, str)
        assert "encountered an error" in committed_event.content.description

    async def test_interrupts_in_flight_generation_commits_partial(self) -> None:
        """Interrupt mid-stream should commit partial text with status=interrupted."""
        stream_started = asyncio.Event()
        allow_continue = asyncio.Event()

        class _BlockingProvider:
            def __init__(self) -> None:
                self.abort = MagicMock(side_effect=lambda: allow_continue.set())

            async def stream(
                self,
                messages: list[ModelMessage],
                config: ArtificerModelConfig,
            ) -> AsyncIterator[StreamChunk]:
                yield StreamChunk(text="partial", done=False)
                stream_started.set()
                # Block until aborted
                await allow_continue.wait()
                yield StreamChunk(text=" more", done=False)
                yield StreamChunk(text="", done=True)

        provider = _BlockingProvider()
        runtime, _, _, event_store = create_runtime(provider)
        runtime.register_artificer(ENTITY_ID, PERSONA, MODEL_CONFIG)

        dispatch_task = asyncio.ensure_future(runtime.handle_dispatch(make_dispatch()))

        # Wait for stream to emit first chunk
        await stream_started.wait()

        # Interrupt
        await runtime.interrupt(CONV_ID)

        # Wait for dispatch to settle
        try:
            await dispatch_task
        except Exception:
            pass

        # Should have committed partial text with interrupted status
        committed_calls = event_store.append_event.call_args_list
        interrupted_commits = [
            call for call in committed_calls if call[0][0].status == "interrupted"
        ]
        assert len(interrupted_commits) >= 1

    async def test_is_processing_returns_correct_state(self) -> None:
        resolve_stream: asyncio.Event = asyncio.Event()

        class _BlockingProvider:
            def __init__(self) -> None:
                self.abort = MagicMock()

            async def stream(
                self,
                messages: list[ModelMessage],
                config: ArtificerModelConfig,
            ) -> AsyncIterator[StreamChunk]:
                yield StreamChunk(text="hi", done=False)
                await resolve_stream.wait()
                yield StreamChunk(text="", done=True)

        provider = _BlockingProvider()
        runtime, _, _, _ = create_runtime(provider)
        runtime.register_artificer(ENTITY_ID, PERSONA, MODEL_CONFIG)

        assert runtime.is_processing(CONV_ID) is False

        p = asyncio.ensure_future(runtime.handle_dispatch(make_dispatch()))
        await asyncio.sleep(0.02)

        assert runtime.is_processing(CONV_ID) is True

        resolve_stream.set()
        await p

        assert runtime.is_processing(CONV_ID) is False

    async def test_concurrent_invocation_limit_enforced_via_queue(self) -> None:
        provider = make_mock_model_provider(["ok"])
        runtime, _, _, event_store = create_runtime(provider, max_concurrent_invocations=1)
        runtime.register_artificer(ENTITY_ID, PERSONA, MODEL_CONFIG)

        entity2 = EntityId("entity-2")
        runtime.register_artificer(
            entity2,
            ArtificerPersona(
                name="Bot2",
                role="analyst",
                phase="Discovery",
                system_prompt="You are Bot2.",
            ),
            MODEL_CONFIG,
        )

        dispatch1 = make_dispatch()
        dispatch2 = DispatchPayload(
            conversation_id=ConversationId("conv-2"),
            entity_id=entity2,
            batch=SealedBatch(
                batch_id=BatchId("batch-2"),
                conversation_id=ConversationId("conv-2"),
                author_id=EntityId("entity-alice"),
                event_ids=[],
                sealed_at=1_000_000.0,
                message_count=1,
            ),
            lock=LockState(
                conversation_id=ConversationId("conv-2"),
                holder_id=entity2,
                acquired_at=1_000_000.0,
                ttl_ms=30_000,
                estimated_ms=20_000,
                expires_at=1_030_000.0,
                metadata={},
            ),
            ai_depth_counter=0,
            trace_id="trace-2",
        )

        # Both dispatches should complete (queue handles serialization)
        await asyncio.gather(
            runtime.handle_dispatch(dispatch1),
            runtime.handle_dispatch(dispatch2),
        )

        # Both should have committed
        assert event_store.append_event.call_count == 2

    async def test_throws_for_unregistered_artificer(self) -> None:
        provider = make_mock_model_provider(["ok"])
        runtime, _, _, _ = create_runtime(provider)
        # Don't register -- should fail

        with pytest.raises(ValueError, match="not registered"):
            await runtime.handle_dispatch(make_dispatch())
