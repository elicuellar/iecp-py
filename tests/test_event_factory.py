import re

from iecp_core import (
    ActionContent,
    AttentionContent,
    BatchId,
    ConversationId,
    DecisionContent,
    EntityId,
    EventId,
    HandoffContent,
    MessageContent,
    SystemContent,
)
from iecp_core.events import (
    create_action_event,
    create_attention_event,
    create_decision_event,
    create_handoff_event,
    create_message_event,
    create_system_event,
)

CONV_ID = ConversationId("conv-1")
AUTHOR_ID = EntityId("entity-1")
ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def test_create_message_event_defaults():
    event = create_message_event(CONV_ID, AUTHOR_ID, "human", "Hello")
    assert event.type == "message"
    assert event.conversation_id == CONV_ID
    assert event.author_id == AUTHOR_ID
    assert event.author_type == "human"
    assert isinstance(event.content, MessageContent)
    assert event.content.text == "Hello"
    assert event.content.format == "plain"
    assert event.content.mentions == []
    assert event.is_continuation is False
    assert event.is_complete is True
    assert event.ai_depth_counter == 0
    assert event.status == "active"
    assert event.metadata == {}
    assert event.parent_id is None
    assert event.batch_id is None
    assert ULID_RE.match(event.id)


def test_create_message_event_markdown_and_mentions():
    mentions = [EntityId("entity-2"), EntityId("entity-3")]
    event = create_message_event(
        CONV_ID, AUTHOR_ID, "artificer", "**bold**",
        format="markdown", mentions=mentions,
    )
    assert isinstance(event.content, MessageContent)
    assert event.content.format == "markdown"
    assert event.content.mentions == mentions


def test_continuation_and_depth_counter():
    event = create_message_event(
        CONV_ID, AUTHOR_ID, "artificer", "Continuing...",
        is_continuation=True, is_complete=False, ai_depth_counter=2,
    )
    assert event.is_continuation is True
    assert event.is_complete is False
    assert event.ai_depth_counter == 2


def test_create_action_event():
    event = create_action_event(
        CONV_ID, EntityId("daemon-1"), "daemon",
        action_type="tool_call", description="Running query",
        result="42 rows", action_status="completed",
    )
    assert event.type == "action"
    assert event.author_type == "daemon"
    assert isinstance(event.content, ActionContent)
    assert event.content.action_type == "tool_call"
    assert event.content.description == "Running query"
    assert event.content.result == "42 rows"
    assert event.content.status == "completed"


def test_create_system_event():
    event = create_system_event(
        CONV_ID, "entity_joined", "User joined the conversation",
        data={"entity_id": "entity-1"},
    )
    assert event.type == "system"
    assert event.author_id == EntityId("system")
    assert event.author_type == "system"
    assert isinstance(event.content, SystemContent)
    assert event.content.system_event == "entity_joined"
    assert event.content.data == {"entity_id": "entity-1"}


def test_create_attention_event_with_optional_fields():
    ref = EventId("ref-event")
    event = create_attention_event(
        CONV_ID, AUTHOR_ID, "human", "urgent",
        utterance_ref=ref, note="Please look at this",
    )
    assert event.type == "attention"
    assert isinstance(event.content, AttentionContent)
    assert event.content.signal == "urgent"
    assert event.content.utterance_ref == ref
    assert event.content.note == "Please look at this"


def test_create_attention_event_without_optional_fields():
    event = create_attention_event(CONV_ID, AUTHOR_ID, "human", "ping")
    assert isinstance(event.content, AttentionContent)
    assert event.content.utterance_ref is None
    assert event.content.note is None


def test_create_decision_event_affirmed():
    event = create_decision_event(
        CONV_ID, AUTHOR_ID, "human",
        summary="We decided X",
        proposed_by=EntityId("entity-2"),
        affirmed_by=[EntityId("entity-1")],
        context_events=[EventId("evt-1")],
        decision_status="affirmed",
    )
    assert event.type == "decision"
    assert isinstance(event.content, DecisionContent)
    assert event.content.summary == "We decided X"
    assert event.content.status == "affirmed"
    assert event.content.affirmed_by == [EntityId("entity-1")]
    assert event.content.context_events == [EventId("evt-1")]


def test_create_decision_event_defaults_to_proposed():
    event = create_decision_event(
        CONV_ID, AUTHOR_ID, "human",
        summary="Proposal", proposed_by=AUTHOR_ID,
    )
    assert isinstance(event.content, DecisionContent)
    assert event.content.status == "proposed"


def test_create_handoff_event():
    event = create_handoff_event(
        CONV_ID, AUTHOR_ID, "artificer",
        from_entity=EntityId("ai-1"), to_entity=EntityId("ai-2"),
        reason="Specialization needed", context_summary="Discussion about X",
        source_event=EventId("src-event"),
    )
    assert event.type == "handoff"
    assert isinstance(event.content, HandoffContent)
    assert event.content.from_entity == EntityId("ai-1")
    assert event.content.to_entity == EntityId("ai-2")
    assert event.content.reason == "Specialization needed"
    assert event.content.source_event == EventId("src-event")


def test_unique_event_ids():
    ids = set()
    for _ in range(50):
        event = create_message_event(CONV_ID, AUTHOR_ID, "human", "msg")
        ids.add(event.id)
    assert len(ids) == 50


def test_preserves_parent_and_batch_id():
    parent = EventId("parent-123")
    batch = BatchId("batch-456")
    event = create_message_event(
        CONV_ID, AUTHOR_ID, "human", "msg",
        parent_id=parent, batch_id=batch,
    )
    assert event.parent_id == parent
    assert event.batch_id == batch
