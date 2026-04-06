"""HandoffManager -- SS20 of the specification.

Manages entity-to-entity handoffs and human escalations.

Handoff rules (SS20.2):
- Only AI entities can initiate handoffs
- Handoffs override routing (target entity gets priority)
- Handoffs expire after configured time OR after target processes next batch
- Chain depth limit: A->B->C (depth 3) -> further handoffs rejected
- Handoffs are visible to all participants

Escalation rules (SS20.3):
- Escalations suppress ALL AI dispatch until human responds
- Any human can resolve an escalation
- Unresolved escalations included in context payloads
"""

from __future__ import annotations

import time
from typing import Optional

from ..types.entity import EntityId
from ..types.event import ConversationId, EventId
from .types import (
    ActiveEscalation,
    ActiveHandoff,
    EscalationRequires,
    HandoffManagerConfig,
    DEFAULT_HANDOFF_MANAGER_CONFIG,
)


class HandoffManager:
    """Manages handoffs and escalations."""

    def __init__(self, config: Optional[HandoffManagerConfig] = None) -> None:
        self._config = config or DEFAULT_HANDOFF_MANAGER_CONFIG
        self._handoffs: dict[ConversationId, ActiveHandoff] = {}
        self._chain_depths: dict[ConversationId, int] = {}
        self._escalations: dict[ConversationId, ActiveEscalation] = {}
        self._destroyed = False

    def handoff(
        self,
        *,
        event_id: EventId,
        conversation_id: ConversationId,
        from_entity: EntityId,
        to_entity: EntityId,
        reason: str,
        context_summary: str,
        source_event: EventId,
    ) -> dict:
        """Record a handoff. Returns dict with success, handoff, and/or error."""
        if self._destroyed:
            return {"success": False, "error": "Manager destroyed"}

        # Check chain depth limit
        current_depth = self._chain_depths.get(conversation_id, 0)
        new_depth = current_depth + 1

        if new_depth > self._config.max_chain_depth:
            return {
                "success": False,
                "error": (
                    f"Handoff chain depth limit exceeded "
                    f"(max: {self._config.max_chain_depth}, current: {current_depth})"
                ),
            }

        # Check if existing handoff has expired
        existing = self._handoffs.get(conversation_id)
        if existing is not None and existing.expires_at <= time.time() * 1000:
            del self._handoffs[conversation_id]

        now = time.time() * 1000
        active_handoff = ActiveHandoff(
            event_id=event_id,
            conversation_id=conversation_id,
            from_entity=from_entity,
            to_entity=to_entity,
            reason=reason,
            context_summary=context_summary,
            source_event=source_event,
            created_at=now,
            expires_at=now + self._config.handoff_expiry_ms,
            chain_depth=new_depth,
        )

        self._handoffs[conversation_id] = active_handoff
        self._chain_depths[conversation_id] = new_depth

        return {"success": True, "handoff": active_handoff}

    def get_active_handoff(
        self, conversation_id: ConversationId
    ) -> Optional[ActiveHandoff]:
        """Get active (non-expired) handoff for a conversation."""
        handoff = self._handoffs.get(conversation_id)
        if handoff is None:
            return None
        if handoff.expires_at <= time.time() * 1000:
            del self._handoffs[conversation_id]
            return None
        return handoff

    def resolve_handoff(self, conversation_id: ConversationId) -> None:
        """Resolve handoff: clears active handoff and resets chain depth."""
        self._handoffs.pop(conversation_id, None)
        self._chain_depths.pop(conversation_id, None)

    def escalate(
        self,
        *,
        event_id: EventId,
        conversation_id: ConversationId,
        entity_id: EntityId,
        reason: str,
        requires: EscalationRequires,
        context_summary: str,
        source_event: EventId,
    ) -> ActiveEscalation:
        """Record an escalation."""
        escalation = ActiveEscalation(
            event_id=event_id,
            conversation_id=conversation_id,
            entity_id=entity_id,
            reason=reason,
            requires=requires,
            context_summary=context_summary,
            source_event=source_event,
            created_at=time.time() * 1000,
            resolved=False,
        )
        self._escalations[conversation_id] = escalation
        return escalation

    def get_active_escalation(
        self, conversation_id: ConversationId
    ) -> Optional[ActiveEscalation]:
        """Get active escalation for a conversation."""
        esc = self._escalations.get(conversation_id)
        if esc is None or esc.resolved:
            return None
        return esc

    def resolve_escalation(
        self, conversation_id: ConversationId, resolved_by: EntityId
    ) -> None:
        """Resolve escalation (human responded)."""
        esc = self._escalations.get(conversation_id)
        if esc is not None and not esc.resolved:
            esc.resolved = True
            esc.resolved_by = resolved_by

    def is_escalation_active(self, conversation_id: ConversationId) -> bool:
        """Check if an escalation is active for a conversation."""
        esc = self._escalations.get(conversation_id)
        return esc is not None and not esc.resolved

    def get_chain_depth(self, conversation_id: ConversationId) -> int:
        """Get current handoff chain depth for a conversation."""
        return self._chain_depths.get(conversation_id, 0)

    def destroy(self) -> None:
        """Destroy the manager -- clear all state."""
        self._destroyed = True
        self._handoffs.clear()
        self._chain_depths.clear()
        self._escalations.clear()
