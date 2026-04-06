"""Artificer Runtime -- §11 of the IECP specification.

The main runtime class that handles the full dispatch lifecycle:
context assembly -> model streaming -> output filtering -> event commit.

Emits events for the integration layer to wire to the WebSocket gateway.
Does NOT directly call the gateway -- that's Phase 10's job.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from ..context.context_builder import BuildContextParams, ContextBuilder
from ..context.types import ContextPayload
from ..events.event_factory import create_message_event, create_system_event
from ..orchestrator.types import DispatchPayload
from ..types.entity import EntityId
from ..types.event import ConversationId, Event
from .context_converter import context_to_messages
from .dispatch_queue import DispatchQueue
from .output_filter import OutputFilter
from .types import (
    DEFAULT_ARTIFICER_RUNTIME_CONFIG,
    ArtificerModelConfig,
    ArtificerPersona,
    ArtificerRuntimeConfig,
    InterruptedEvent,
    MessageCommittedEvent,
    ModelProvider,
    RuntimeErrorEvent,
    StreamChunkEvent,
)


# -- Registration Entry -------------------------------------------------------


@dataclass
class _ArtificerRegistration:
    entity_id: EntityId
    persona: ArtificerPersona
    model_config: ArtificerModelConfig


# -- Active Invocation --------------------------------------------------------


@dataclass
class _ActiveInvocation:
    conversation_id: ConversationId
    entity_id: EntityId
    partial_text: str = ""
    chunk_index: int = 0
    aborted: bool = False


# -- Interfaces ---------------------------------------------------------------


class FloorLockLike(Protocol):
    """Minimal interface for the floor lock."""

    def release(
        self,
        conversation_id: ConversationId,
        entity_id: EntityId,
        reason: str,
    ) -> bool: ...


class EventStoreLike(Protocol):
    """Minimal async interface for the event store."""

    async def append_event(self, event: Event) -> Event: ...


# -- ArtificerRuntime ---------------------------------------------------------


class ArtificerRuntime:
    """Runtime that handles the full artificer dispatch lifecycle."""

    def __init__(
        self,
        model_provider: ModelProvider,
        context_builder: ContextBuilder,
        output_filter: OutputFilter,
        floor_lock: FloorLockLike,
        event_store: EventStoreLike,
        config: ArtificerRuntimeConfig | None = None,
    ) -> None:
        self._model_provider = model_provider
        self._context_builder = context_builder
        self._output_filter = output_filter
        self._floor_lock = floor_lock
        self._event_store = event_store
        self._config = config or DEFAULT_ARTIFICER_RUNTIME_CONFIG

        self._dispatch_queue = DispatchQueue(self._config.max_concurrent_invocations)
        self._registrations: dict[str, _ArtificerRegistration] = {}
        self._active_invocations: dict[str, _ActiveInvocation] = {}
        self._listeners: dict[str, set[Callable[..., Any]]] = {}

    # -- Event Emitter --------------------------------------------------------

    def on(self, event: str, listener: Callable[..., Any]) -> None:
        """Register an event listener."""
        if event not in self._listeners:
            self._listeners[event] = set()
        self._listeners[event].add(listener)

    def _emit(self, event: str, payload: Any) -> None:
        listeners = self._listeners.get(event)
        if listeners:
            for cb in listeners:
                cb(payload)

    # -- Public API -----------------------------------------------------------

    def register_artificer(
        self,
        entity_id: EntityId,
        persona: ArtificerPersona,
        model_config: ArtificerModelConfig,
    ) -> None:
        """Register an artificer with its persona and model config.

        Must be called before the artificer can be dispatched to.
        """
        self._registrations[entity_id] = _ArtificerRegistration(
            entity_id=entity_id,
            persona=persona,
            model_config=model_config,
        )

    async def handle_dispatch(self, dispatch: DispatchPayload) -> None:
        """Handle a dispatch from the orchestrator.

        Enqueues via the DispatchQueue to respect concurrency limits.
        """
        await self._dispatch_queue.enqueue(dispatch, self._process_dispatch)

    async def interrupt(self, conversation_id: ConversationId) -> None:
        """Interrupt an in-flight generation.

        Aborts the model, commits partial output, releases the lock.
        """
        invocation = self._active_invocations.get(conversation_id)
        if not invocation:
            return

        invocation.aborted = True
        self._model_provider.abort()

        # Commit partial output if any
        if invocation.partial_text.strip():
            event = create_message_event(
                conversation_id=conversation_id,
                author_id=invocation.entity_id,
                author_type="artificer",
                text=invocation.partial_text,
            )
            # Patch status to 'interrupted'
            event = event.model_copy(update={"status": "interrupted"})
            await self._event_store.append_event(event)

            self._emit(
                "message_committed",
                MessageCommittedEvent(
                    conversation_id=conversation_id,
                    entity_id=invocation.entity_id,
                    event=event,
                ),
            )

        self._emit(
            "interrupted",
            InterruptedEvent(
                conversation_id=conversation_id,
                entity_id=invocation.entity_id,
                partial_text=invocation.partial_text,
            ),
        )

        self._floor_lock.release(conversation_id, invocation.entity_id, "human_interrupt")
        del self._active_invocations[conversation_id]

    def is_processing(self, conversation_id: ConversationId) -> bool:
        """Check if an artificer is currently generating in a conversation."""
        return conversation_id in self._active_invocations

    def get_concurrent_count(self) -> int:
        """Get the number of concurrent invocations."""
        return len(self._active_invocations)

    def get_queue_stats(self) -> dict[str, int]:
        """Get dispatch queue statistics."""
        stats = self._dispatch_queue.get_stats()
        return {"active": stats.active, "queued": stats.queued}

    # -- Private: Main Dispatch Processing ------------------------------------

    async def _process_dispatch(self, dispatch: DispatchPayload) -> None:
        entity_id = dispatch.entity_id
        registration = self._registrations.get(entity_id)

        if not registration:
            raise ValueError(f"Artificer {entity_id} is not registered")

        persona = registration.persona
        model_config = registration.model_config
        conversation_id = dispatch.conversation_id

        invocation = _ActiveInvocation(
            conversation_id=conversation_id,
            entity_id=entity_id,
        )
        self._active_invocations[conversation_id] = invocation

        try:
            await self._process_with_retry(dispatch, invocation, persona, model_config)
        finally:
            self._active_invocations.pop(conversation_id, None)

    async def _process_with_retry(
        self,
        dispatch: DispatchPayload,
        invocation: _ActiveInvocation,
        persona: ArtificerPersona,
        model_config: ArtificerModelConfig,
    ) -> None:
        last_error: Exception | None = None
        conversation_id = dispatch.conversation_id
        entity_id = dispatch.entity_id

        for attempt in range(self._config.max_retries + 1):
            if invocation.aborted:
                return

            try:
                # 1. Build context
                build_params = BuildContextParams(
                    dispatch=dispatch,
                    system_prompt=persona.system_prompt,
                )
                context_payload = await self._context_builder.build_context(build_params)

                # 2. Convert to model messages
                messages = context_to_messages(context_payload, persona)

                # 3. Stream completion
                invocation.partial_text = ""
                invocation.chunk_index = 0

                async for chunk in self._model_provider.stream(messages, model_config):
                    if invocation.aborted:
                        return

                    if chunk.text:
                        invocation.partial_text += chunk.text
                        invocation.chunk_index += 1

                        self._emit(
                            "stream_chunk",
                            StreamChunkEvent(
                                conversation_id=conversation_id,
                                entity_id=entity_id,
                                text=chunk.text,
                                chunk_index=invocation.chunk_index,
                            ),
                        )

                    if chunk.done:
                        break

                # 4. Run output filter
                filter_result = self._output_filter.check(invocation.partial_text, persona)
                if filter_result is not None:
                    # Output rejected -- post system event
                    system_event = create_system_event(
                        conversation_id=conversation_id,
                        system_event="output_filtered",
                        description=(
                            f"{persona.name}'s response was filtered: {filter_result}"
                        ),
                    )
                    await self._event_store.append_event(system_event)
                    self._floor_lock.release(conversation_id, entity_id, "commit")
                    return

                # 5. Commit message event
                message_event = create_message_event(
                    conversation_id=conversation_id,
                    author_id=entity_id,
                    author_type="artificer",
                    text=invocation.partial_text,
                    batch_id=dispatch.batch.batch_id,
                    ai_depth_counter=dispatch.ai_depth_counter + 1,
                )
                await self._event_store.append_event(message_event)

                self._emit(
                    "message_committed",
                    MessageCommittedEvent(
                        conversation_id=conversation_id,
                        entity_id=entity_id,
                        event=message_event,
                    ),
                )

                # 6. Release the lock
                self._floor_lock.release(conversation_id, entity_id, "commit")
                return

            except Exception as exc:
                last_error = exc

                self._emit(
                    "error",
                    RuntimeErrorEvent(
                        conversation_id=conversation_id,
                        entity_id=entity_id,
                        error=str(exc),
                        retry_count=attempt + 1,
                    ),
                )

                if attempt < self._config.max_retries:
                    # Exponential backoff: base -> base*2 -> base*4
                    delay_ms = self._config.retry_base_delay_ms * (2 ** attempt)
                    await asyncio.sleep(delay_ms / 1000.0)

        # All retries exhausted -- post system event and release lock
        fail_event = create_system_event(
            conversation_id=conversation_id,
            system_event="artificer_error",
            description=f"{persona.name} encountered an error and could not respond.",
            data={"error": str(last_error) if last_error else None},
        )
        await self._event_store.append_event(fail_event)
        self._floor_lock.release(conversation_id, entity_id, "commit")
