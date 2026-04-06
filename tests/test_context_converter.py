"""Context Converter tests -- Phase 6: Artificer Runtime (§11 of the IECP spec)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from iecp_core.artificer import ArtificerPersona, context_to_messages
from iecp_core.context.types import ContextPayload, ParticipantSummary
from iecp_core.types.entity import EntityId
from iecp_core.types.event import (
    BatchId,
    ConversationId,
    DecisionContent,
    Event,
    EventId,
    HandoffContent,
    MessageContent,
    SystemContent,
)

ARTIFICER_ID = EntityId("entity-meina")
HUMAN_ID = EntityId("entity-alice")
CONV_ID = ConversationId("conv-1")

PERSONA = ArtificerPersona(
    name="Meina",
    role="analyst",
    phase="Discovery",
    system_prompt="You are Meina, a Discovery analyst.",
)

_NOW = datetime.now(timezone.utc).isoformat()


def make_payload(**overrides) -> ContextPayload:
    base = dict(
        conversation_id=CONV_ID,
        recipient_id=ARTIFICER_ID,
        unread_messages=[],
        recent_history=[],
        conversation_summary=None,
        participants=[
            ParticipantSummary(
                entity_id=ARTIFICER_ID,
                display_name="Meina",
                entity_type="artificer",
                capabilities=None,
                lifecycle_status="active",
            ),
            ParticipantSummary(
                entity_id=HUMAN_ID,
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
    base.update(overrides)
    return ContextPayload(**base)


def make_message_event(
    author_id: EntityId,
    author_type: str,
    text: str,
) -> Event:
    return Event(
        id=EventId("evt-1"),
        conversation_id=CONV_ID,
        author_id=author_id,
        author_type=author_type,
        type="message",
        content=MessageContent(text=text, format="plain", mentions=[]),
        is_continuation=False,
        is_complete=True,
        ai_depth_counter=0,
        status="active",
        created_at=_NOW,
        metadata={},
    )


def make_system_event(description: str) -> Event:
    return Event(
        id=EventId("evt-sys"),
        conversation_id=CONV_ID,
        author_id=EntityId("system"),
        author_type="system",
        type="system",
        content=SystemContent(system_event="info", description=description, data={}),
        is_continuation=False,
        is_complete=True,
        ai_depth_counter=0,
        status="active",
        created_at=_NOW,
        metadata={},
    )


class TestContextToMessages:
    def test_converts_simple_payload_system_and_user(self) -> None:
        payload = make_payload(
            unread_messages=[make_message_event(HUMAN_ID, "human", "Hello Meina!")]
        )
        messages = context_to_messages(payload, PERSONA)

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert "You are Meina" in messages[0].content
        assert messages[1].role == "user"
        assert "[Alice]:" in messages[1].content
        assert "Hello Meina!" in messages[1].content

    def test_maps_artificers_own_messages_to_assistant_role(self) -> None:
        payload = make_payload(
            recent_history=[make_message_event(ARTIFICER_ID, "artificer", "I analyzed the data.")]
        )
        messages = context_to_messages(payload, PERSONA)

        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "I analyzed the data."

    def test_maps_other_entities_to_user_role_with_prefix(self) -> None:
        payload = make_payload(
            unread_messages=[make_message_event(HUMAN_ID, "human", "What do you think?")]
        )
        messages = context_to_messages(payload, PERSONA)

        user_msgs = [m for m in messages if m.role == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "[Alice]: What do you think?"

    def test_includes_active_decisions_in_system_message(self) -> None:
        decision_event = Event(
            id=EventId("evt-dec"),
            conversation_id=CONV_ID,
            author_id=HUMAN_ID,
            author_type="human",
            type="decision",
            content=DecisionContent(
                summary="Use React for the frontend",
                proposed_by=HUMAN_ID,
                affirmed_by=[],
                context_events=[],
                status="proposed",
            ),
            is_continuation=False,
            is_complete=True,
            ai_depth_counter=0,
            status="active",
            created_at=_NOW,
            metadata={},
        )
        payload = make_payload(active_decisions=[decision_event])
        messages = context_to_messages(payload, PERSONA)

        system_msg = messages[0]
        assert "Active Decisions" in system_msg.content
        assert "Use React for the frontend" in system_msg.content

    def test_includes_participant_manifest_in_system_message(self) -> None:
        payload = make_payload()
        messages = context_to_messages(payload, PERSONA)

        system_msg = messages[0]
        assert "Participants" in system_msg.content
        assert "Meina" in system_msg.content
        assert "Alice" in system_msg.content
        assert "artificer" in system_msg.content
        assert "human" in system_msg.content

    def test_handles_empty_recent_history(self) -> None:
        payload = make_payload(
            recent_history=[],
            unread_messages=[make_message_event(HUMAN_ID, "human", "Hi")],
        )
        messages = context_to_messages(payload, PERSONA)
        # system + 1 unread
        assert len(messages) == 2

    def test_handles_empty_unread_messages(self) -> None:
        payload = make_payload(
            recent_history=[make_message_event(HUMAN_ID, "human", "Earlier message")],
            unread_messages=[],
        )
        messages = context_to_messages(payload, PERSONA)
        # system + 1 history
        assert len(messages) == 2

    def test_maps_system_events_to_system_role(self) -> None:
        payload = make_payload(
            unread_messages=[make_system_event("A user joined the conversation")]
        )
        messages = context_to_messages(payload, PERSONA)
        sys_msgs = [m for m in messages if m.role == "system" and "joined" in m.content]
        assert len(sys_msgs) == 1

    def test_includes_pending_handoffs_in_system_message(self) -> None:
        handoff_event = Event(
            id=EventId("evt-hoff"),
            conversation_id=CONV_ID,
            author_id=HUMAN_ID,
            author_type="human",
            type="handoff",
            content=HandoffContent(
                from_entity=HUMAN_ID,
                to_entity=ARTIFICER_ID,
                reason="Need deeper analysis",
                context_summary="User wants performance metrics reviewed",
                source_event=EventId("evt-1"),
            ),
            is_continuation=False,
            is_complete=True,
            ai_depth_counter=0,
            status="active",
            created_at=_NOW,
            metadata={},
        )
        payload = make_payload(pending_handoffs=[handoff_event])
        messages = context_to_messages(payload, PERSONA)

        system_msg = messages[0]
        assert "Pending Handoffs" in system_msg.content
        assert "Need deeper analysis" in system_msg.content
