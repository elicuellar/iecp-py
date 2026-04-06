"""Context-to-Messages Converter -- §11 of the IECP specification.

Converts a ContextPayload (§9.1) into list[ModelMessage] for LLM consumption.
Maps conversation events to the system/user/assistant roles expected by
OpenAI-compatible APIs.
"""

from __future__ import annotations

from ..context.types import ContextPayload, ParticipantSummary
from ..types.event import DecisionContent, Event, HandoffContent, MessageContent, SystemContent
from .types import ArtificerPersona, ModelMessage


# -- Helpers ------------------------------------------------------------------


def _event_to_text(event: Event) -> str:
    if event.type == "message":
        assert isinstance(event.content, MessageContent)
        return event.content.text
    elif event.type == "system":
        assert isinstance(event.content, SystemContent)
        return f"[System: {event.content.description}]"
    elif event.type == "decision":
        assert isinstance(event.content, DecisionContent)
        return f"[Decision ({event.content.status}): {event.content.summary}]"
    elif event.type == "handoff":
        assert isinstance(event.content, HandoffContent)
        return (
            f"[Handoff from {event.content.from_entity} to "
            f"{event.content.to_entity}: {event.content.reason}]"
        )
    else:
        return f"[{event.type} event]"


def _find_participant_name(payload: ContextPayload, entity_id: str) -> str:
    for p in payload.participants:
        if p.entity_id == entity_id:
            return p.display_name
    return "Unknown"


# -- Build System Message -----------------------------------------------------


def _build_system_message(payload: ContextPayload, persona: ArtificerPersona) -> str:
    parts: list[str] = [persona.system_prompt]

    # Participant manifest
    if payload.participants:
        manifest_lines = [
            f"- {p.display_name} ({p.entity_type}, {p.lifecycle_status})"
            for p in payload.participants
        ]
        parts.append("\nParticipants:\n" + "\n".join(manifest_lines))

    # Active decisions
    if payload.active_decisions:
        decision_lines = []
        for d in payload.active_decisions:
            if isinstance(d.content, DecisionContent):
                decision_lines.append(
                    f"- [{d.content.status}] {d.content.summary}"
                )
        if decision_lines:
            parts.append("\nActive Decisions:\n" + "\n".join(decision_lines))

    # Pending handoffs
    if payload.pending_handoffs:
        handoff_lines = []
        for h in payload.pending_handoffs:
            if isinstance(h.content, HandoffContent):
                handoff_lines.append(
                    f"- From {h.content.from_entity}: {h.content.reason}\n"
                    f"  Context: {h.content.context_summary}"
                )
        if handoff_lines:
            parts.append("\nPending Handoffs to You:\n" + "\n".join(handoff_lines))

    return "\n".join(parts)


# -- Event to ModelMessage ----------------------------------------------------


def _event_to_model_message(event: Event, payload: ContextPayload) -> ModelMessage:
    recipient_id = payload.recipient_id
    author_id = event.author_id

    # System events → system role
    if event.author_type == "system" or event.type == "system":
        return ModelMessage(role="system", content=_event_to_text(event))

    # Artificer's own messages → assistant role
    if author_id == recipient_id:
        return ModelMessage(role="assistant", content=_event_to_text(event))

    # All other entities → user role with name prefix
    name = _find_participant_name(payload, author_id)
    return ModelMessage(role="user", content=f"[{name}]: {_event_to_text(event)}")


# -- Main Converter -----------------------------------------------------------


def context_to_messages(
    payload: ContextPayload,
    persona: ArtificerPersona,
) -> list[ModelMessage]:
    """Convert a ContextPayload into list[ModelMessage] for the LLM.

    Rules:
    - System message: persona.system_prompt + participant manifest + active
      decisions + pending handoffs
    - The artificer's own previous messages → role: 'assistant'
    - All other entities' messages → role: 'user' with `[Name]: ` prefix
    - System events → role: 'system'
    """
    messages: list[ModelMessage] = []

    # 1. System message
    messages.append(
        ModelMessage(role="system", content=_build_system_message(payload, persona))
    )

    # 2. Recent history as user/assistant messages
    for event in payload.recent_history:
        messages.append(_event_to_model_message(event, payload))

    # 3. Unread messages
    for event in payload.unread_messages:
        messages.append(_event_to_model_message(event, payload))

    return messages
