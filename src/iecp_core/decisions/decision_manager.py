"""DecisionManager -- SS19 of the specification.

Manages the lifecycle of decisions: propose -> affirm/reject/supersede.

Rules (SS19.4):
- Any entity can propose (human, artificer, daemon)
- Decisions require human affirmation by default
- Active decisions = proposed + affirmed (not rejected/superseded)
- Superseding links new to old, marks old as superseded
- Decisions do NOT require floor lock
"""

from __future__ import annotations

import time
from typing import Optional

from ..types.entity import EntityId
from ..types.event import ConversationId, EventId
from .types import (
    Decision,
    DecisionManagerConfig,
    DEFAULT_DECISION_MANAGER_CONFIG,
)


class DecisionManager:
    """Manages decisions within conversations."""

    def __init__(self, config: Optional[DecisionManagerConfig] = None) -> None:
        self._config = config or DEFAULT_DECISION_MANAGER_CONFIG
        self._decisions: dict[EventId, Decision] = {}

    def propose(
        self,
        *,
        event_id: EventId,
        conversation_id: ConversationId,
        summary: str,
        proposed_by: EntityId,
        context_events: list[EventId],
    ) -> Decision:
        """Record a new decision proposal."""
        decision = Decision(
            event_id=event_id,
            conversation_id=conversation_id,
            summary=summary,
            proposed_by=proposed_by,
            proposed_at=time.time() * 1000,
            context_events=list(context_events),
            affirmed_by=[],
            rejected_by=[],
            status="proposed",
            superseded_by=None,
        )
        self._decisions[event_id] = decision
        return decision

    def affirm(
        self,
        event_id: EventId,
        affirmer_id: EntityId,
        is_human: bool = True,
    ) -> Optional[Decision]:
        """Affirm a decision. Returns None if not found or not allowed."""
        decision = self._decisions.get(event_id)
        if decision is None:
            return None

        # If human affirmation required and affirmer is not human, reject
        if self._config.require_human_affirmation and not is_human:
            return None

        # Track the affirmer (no duplicates)
        if affirmer_id not in decision.affirmed_by:
            decision.affirmed_by.append(affirmer_id)

        decision.status = "affirmed"
        return decision

    def reject(
        self, event_id: EventId, rejecter_id: EntityId
    ) -> Optional[Decision]:
        """Reject a decision. Returns None if not found."""
        decision = self._decisions.get(event_id)
        if decision is None:
            return None

        if rejecter_id not in decision.rejected_by:
            decision.rejected_by.append(rejecter_id)

        decision.status = "rejected"
        return decision

    def supersede(
        self,
        old_event_id: EventId,
        *,
        event_id: EventId,
        conversation_id: ConversationId,
        summary: str,
        proposed_by: EntityId,
        context_events: list[EventId],
    ) -> Optional[dict[str, Decision]]:
        """Supersede a decision with a new one. Returns None if old not found."""
        old_decision = self._decisions.get(old_event_id)
        if old_decision is None:
            return None

        # Mark old as superseded
        old_decision.status = "superseded"
        old_decision.superseded_by = event_id

        # Create the new decision
        new_decision = self.propose(
            event_id=event_id,
            conversation_id=conversation_id,
            summary=summary,
            proposed_by=proposed_by,
            context_events=context_events,
        )

        return {"old": old_decision, "new": new_decision}

    def get_active_decisions(
        self, conversation_id: ConversationId
    ) -> list[Decision]:
        """Get active decisions for a conversation (proposed + affirmed)."""
        return [
            d
            for d in self._decisions.values()
            if d.conversation_id == conversation_id
            and d.status in ("proposed", "affirmed")
        ]

    def get_decision(self, event_id: EventId) -> Optional[Decision]:
        """Get a specific decision by event ID."""
        return self._decisions.get(event_id)

    def get_all_decisions(
        self, conversation_id: ConversationId
    ) -> list[Decision]:
        """Get all decisions for a conversation (including rejected/superseded)."""
        return [
            d
            for d in self._decisions.values()
            if d.conversation_id == conversation_id
        ]
