"""ContextBuilder -- Phase 5 (section 9) of the IECP specification.

Assembles the context payload for an AI entity, respecting
token budgets and priority ordering.

Assembly priority:
1. Mandatory: participants, unread messages, active decisions, pending handoffs
2. Remaining budget -> recent history (most recent first)
3. If still available -> conversation summary

Overflow: oversized individual messages are truncated with retrieval markers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..conversations.conversation_manager import ConversationManager
from ..cursors.cursor_manager import CursorManager
from ..entities.entity_manager import EntityManager
from ..events.event_store import EventStore, ReadEventsOptions
from ..orchestrator.types import DispatchPayload
from ..types.entity import EntityId
from ..types.event import ConversationId, DecisionContent, Event, HandoffContent, MessageContent
from .summarizer import ConversationSummarizer
from .token_estimator import SimpleTokenEstimator
from .types import (
    ContextBuilderConfig,
    ContextPayload,
    DEFAULT_CONTEXT_BUILDER_CONFIG,
    ParticipantSummary,
    TokenEstimator,
)

# -- Truncation Marker --------------------------------------------------------

TRUNCATION_PREFIX_LENGTH = 200


def _truncation_marker(event_id: str) -> str:
    return f"[truncated -- call fetch_message({event_id}) for full content]"


# -- Dependencies -------------------------------------------------------------


@dataclass
class ContextBuilderDeps:
    event_store: EventStore
    entity_manager: EntityManager
    conversation_manager: ConversationManager
    cursor_manager: CursorManager
    token_estimator: TokenEstimator | None = None
    config: ContextBuilderConfig | None = None
    summarizer: ConversationSummarizer | None = None


# -- Build Params -------------------------------------------------------------


@dataclass
class BuildContextParams:
    dispatch: DispatchPayload
    system_prompt: str | None = None
    entity_instructions: str | None = None


# -- ContextBuilder -----------------------------------------------------------


class ContextBuilder:
    def __init__(
        self,
        event_store: EventStore,
        entity_manager: EntityManager,
        conversation_manager: ConversationManager,
        cursor_manager: CursorManager,
        token_estimator: TokenEstimator | None = None,
        config: dict[str, Any] | None = None,
        summarizer: ConversationSummarizer | None = None,
    ) -> None:
        self._event_store = event_store
        self._entity_manager = entity_manager
        self._conversation_manager = conversation_manager
        self._cursor_manager = cursor_manager
        self._estimator: TokenEstimator = token_estimator or SimpleTokenEstimator()

        if config:
            self._config = replace(DEFAULT_CONTEXT_BUILDER_CONFIG, **config)
        else:
            self._config = DEFAULT_CONTEXT_BUILDER_CONFIG

        self._summarizer = summarizer

        # In-memory summary storage (will move to DB in Phase 7)
        self._summaries: dict[ConversationId, str] = {}
        # In-memory counters for summary triggering
        self._message_counters: dict[ConversationId, int] = {}

    async def build_context(self, params: BuildContextParams) -> ContextPayload:
        """Assemble a context payload for the dispatched AI entity."""
        dispatch = params.dispatch
        conversation_id = dispatch.conversation_id
        entity_id = dispatch.entity_id

        total_budget = self._config.default_token_budget
        tokens_used = 0

        # 1. System prompt budget (reserved, not included in payload)
        if params.system_prompt:
            tokens_used += min(
                self._estimator.estimate(params.system_prompt),
                self._config.system_prompt_budget,
            )

        # 2. Participant manifest
        participants = await self._build_participants(conversation_id)
        participant_tokens = len(participants) * self._config.participant_budget_per_entity
        tokens_used += participant_tokens

        # 3. Unread messages
        cursor = await self._cursor_manager.get_cursor(entity_id, conversation_id)
        after_event_id = cursor.cursor_processed if cursor.cursor_processed is not None else None

        if after_event_id is not None:
            unread_result = await self._event_store.read_events(
                conversation_id,
                ReadEventsOptions(after=after_event_id),
            )
        else:
            unread_result = await self._event_store.read_events(conversation_id)

        unread_messages = [
            e for e in unread_result.events
            if e.status in ("active", "edited")
        ]

        # Estimate unread tokens, truncating oversized ones
        unread_tokens = 0
        truncated_unread: list[Event] = []
        for event in unread_messages:
            event_tokens = self._estimator.estimate_event(event)
            if event_tokens > total_budget * 0.5:
                truncated = self._truncate_event(event)
                truncated_unread.append(truncated)
                unread_tokens += self._estimator.estimate_event(truncated)
            else:
                truncated_unread.append(event)
                unread_tokens += event_tokens
        unread_messages = truncated_unread
        tokens_used += unread_tokens

        # 4. Active decisions
        all_events_result = await self._event_store.read_events(conversation_id)
        active_decisions = [
            e for e in all_events_result.events
            if e.type == "decision"
            and e.status == "active"
            and isinstance(e.content, DecisionContent)
            and e.content.status != "superseded"
        ]

        decision_tokens = sum(
            self._estimator.estimate_event(e) for e in active_decisions
        )
        tokens_used += decision_tokens

        # 5. Pending handoffs
        pending_handoffs = [
            e for e in all_events_result.events
            if e.type == "handoff"
            and e.status == "active"
            and isinstance(e.content, HandoffContent)
            and e.content.to_entity == entity_id
        ]

        handoff_tokens = sum(
            self._estimator.estimate_event(e) for e in pending_handoffs
        )
        tokens_used += handoff_tokens

        # 6. Recent history (fill remaining budget)
        remaining_for_history = total_budget - tokens_used - self._config.summary_budget
        recent_history = await self._build_recent_history(
            conversation_id,
            after_event_id,
            max(0, remaining_for_history),
        )

        history_tokens = sum(
            self._estimator.estimate_event(e) for e in recent_history
        )
        tokens_used += history_tokens

        # 7. Conversation summary
        conversation_summary: str | None = None
        remaining_for_summary = total_budget - tokens_used

        if remaining_for_summary > 0:
            conversation_summary = self._summaries.get(conversation_id)
            if conversation_summary is not None:
                summary_tokens = self._estimator.estimate(conversation_summary)
                if summary_tokens <= remaining_for_summary:
                    tokens_used += summary_tokens
                else:
                    conversation_summary = None  # doesn't fit
            # else stays None

        # 8. Trigger summarization if needed
        await self._maybe_trigger_summarization(conversation_id, unread_messages)

        # 9. Recipient context
        entity = await self._entity_manager.get_entity(entity_id)
        your_role = entity.name if entity else "AI Assistant"
        your_capabilities = entity.capabilities.domains if entity and entity.capabilities else []

        return ContextPayload(
            conversation_id=conversation_id,
            recipient_id=entity_id,
            unread_messages=unread_messages,
            recent_history=recent_history,
            conversation_summary=conversation_summary,
            participants=participants,
            response_expected=True,
            batch_id=dispatch.batch.batch_id,
            your_role=your_role,
            your_capabilities=your_capabilities,
            your_instructions=params.entity_instructions,
            active_decisions=active_decisions,
            pending_handoffs=pending_handoffs,
            token_budget=total_budget,
            tokens_used=tokens_used,
        )

    # -- Private Helpers --------------------------------------------------------

    async def _build_participants(
        self, conversation_id: ConversationId
    ) -> list[ParticipantSummary]:
        participants = await self._conversation_manager.get_participants(conversation_id)
        summaries: list[ParticipantSummary] = []

        for p in participants:
            entity = await self._entity_manager.get_entity(p.entity_id)
            summaries.append(
                ParticipantSummary(
                    entity_id=p.entity_id,
                    display_name=entity.name if entity else "Unknown",
                    entity_type=entity.type if entity else "human",
                    capabilities=entity.capabilities if entity else None,
                    lifecycle_status=p.lifecycle_status,
                )
            )

        return summaries

    async def _build_recent_history(
        self,
        conversation_id: ConversationId,
        before_event_id: str | None,
        budget_tokens: int,
    ) -> list[Event]:
        if budget_tokens <= 0:
            return []

        # Read all events, then filter to those before the cursor
        result = await self._event_store.read_events(conversation_id)
        candidates = [
            e for e in result.events
            if e.status in ("active", "edited")
        ]

        if before_event_id:
            candidates = [e for e in candidates if e.id < before_event_id]

        # Take most recent first, up to budget and max count
        reversed_candidates = list(reversed(candidates))
        selected: list[Event] = []
        used = 0

        for event in reversed_candidates:
            if len(selected) >= self._config.recent_history_max_events:
                break
            event_tokens = self._estimator.estimate_event(event)
            if used + event_tokens > budget_tokens:
                break
            selected.append(event)
            used += event_tokens

        # Return in chronological order (oldest first)
        return list(reversed(selected))

    def _truncate_event(self, event: Event) -> Event:
        if event.type != "message":
            return event

        assert isinstance(event.content, MessageContent)
        content = event.content
        prefix = content.text[:TRUNCATION_PREFIX_LENGTH]
        marker = _truncation_marker(event.id)

        new_content = content.model_copy(update={"text": f"{prefix} {marker}"})
        return event.model_copy(update={"content": new_content})

    async def _maybe_trigger_summarization(
        self,
        conversation_id: ConversationId,
        new_messages: list[Event],
    ) -> None:
        if not self._summarizer:
            return

        count = (self._message_counters.get(conversation_id) or 0) + len(new_messages)
        self._message_counters[conversation_id] = count

        if count >= self._config.summary_trigger_messages:
            existing_summary = self._summaries.get(conversation_id)
            summary = await self._summarizer.summarize(
                conversation_id=conversation_id,
                events=new_messages,
                existing_summary=existing_summary,
            )
            self._summaries[conversation_id] = summary
            self._message_counters[conversation_id] = 0

    # -- Test Helpers -----------------------------------------------------------

    def set_summary(self, conversation_id: ConversationId, summary: str) -> None:
        """Set a conversation summary directly (for testing)."""
        self._summaries[conversation_id] = summary

    def get_summary(self, conversation_id: ConversationId) -> str | None:
        """Get the current summary for a conversation."""
        return self._summaries.get(conversation_id)
