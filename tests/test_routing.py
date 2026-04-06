"""Routing Engine tests -- Phase 4: Orchestration.

Pure function tests -- no mocks, no timers, fully deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone

from iecp_core.debounce.types import SealedBatch
from iecp_core.orchestrator.routing import RoutingParams, resolve_routing
from iecp_core.orchestrator.types import DEFAULT_ORCHESTRATOR_CONFIG, OrchestratorConfig
from iecp_core.types.conversation import Conversation, ConversationConfig, Participant
from iecp_core.types.entity import Entity, EntityCapabilities, EntityId
from iecp_core.types.event import (
    BatchId,
    ConversationId,
    Event,
    EventId,
    HandoffContent,
    MessageContent,
)
from iecp_core.utils import generate_id

# -- Helpers -----------------------------------------------------------------

CONV_ID = ConversationId("conv-routing")
HUMAN_A = EntityId("human-a")
AI_A = EntityId("ai-a")
AI_B = EntityId("ai-b")
AI_C = EntityId("ai-c")
DAEMON_A = EntityId("daemon-a")

_NOW = datetime.now(timezone.utc).isoformat()


def _make_entity(
    entity_id: EntityId,
    entity_type: str,
    domains: list[str] | None = None,
) -> Entity:
    return Entity(
        id=entity_id,
        name=str(entity_id),
        type=entity_type,
        capabilities=EntityCapabilities(
            domains=domains or [],
        ),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_entities(
    *defs: tuple[EntityId, str] | tuple[EntityId, str, list[str]],
) -> dict[EntityId, Entity]:
    entities: dict[EntityId, Entity] = {}
    for d in defs:
        entity_id = d[0]
        entity_type = d[1]
        domains = d[2] if len(d) > 2 else None
        entities[entity_id] = _make_entity(entity_id, entity_type, domains)
    return entities


def _make_participant(entity_id: EntityId) -> Participant:
    return Participant(
        entity_id=entity_id,
        conversation_id=CONV_ID,
        role="member",
        lifecycle_status="active",
        joined_at=_NOW,
    )


def _make_participants(entity_ids: list[EntityId]) -> list[Participant]:
    return [_make_participant(eid) for eid in entity_ids]


def _make_conversation() -> Conversation:
    return Conversation(
        id=CONV_ID,
        title="Test Conversation",
        config=ConversationConfig(),
        status="active",
        created_by=HUMAN_A,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_event(
    author_id: EntityId = HUMAN_A,
    text: str = "Hello AI",
    mentions: list[EntityId] | None = None,
) -> Event:
    return Event(
        id=EventId(generate_id()),
        conversation_id=CONV_ID,
        type="message",
        author_id=author_id,
        author_type="human",
        content=MessageContent(
            text=text,
            mentions=mentions or [],
        ),
        created_at=_NOW,
    )


def _make_batch(author_id: EntityId = HUMAN_A) -> SealedBatch:
    return SealedBatch(
        batch_id=BatchId(generate_id()),
        conversation_id=CONV_ID,
        author_id=author_id,
        event_ids=[EventId(generate_id())],
        sealed_at=0.0,
        message_count=1,
    )


def _make_params(
    entities: dict[EntityId, Entity],
    participants: list[Participant],
    events: list[Event] | None = None,
    config: OrchestratorConfig | None = None,
    active_handoff: HandoffContent | None = None,
    last_served: dict[EntityId, float] | None = None,
    batch: SealedBatch | None = None,
) -> RoutingParams:
    return RoutingParams(
        batch=batch or _make_batch(),
        events=events or [_make_event()],
        conversation=_make_conversation(),
        participants=participants,
        entities=entities,
        config=config or OrchestratorConfig(**{
            k: v for k, v in DEFAULT_ORCHESTRATOR_CONFIG.__dict__.items()
        }),
        active_handoff=active_handoff,
        last_served=last_served,
    )


# -- Tests -------------------------------------------------------------------


class TestResolveRouting:
    def test_returns_no_eligible_when_there_are_no_ai_participants(self) -> None:
        entities = _make_entities((HUMAN_A, "human"))
        participants = _make_participants([HUMAN_A])

        result = resolve_routing(_make_params(entities, participants))

        assert result.rule_applied == "no_eligible"
        assert result.selected_entity is None
        assert len(result.eligible_entities) == 0

    def test_selects_mentioned_entity_in_mentioned_only_mode(self) -> None:
        entities = _make_entities(
            (HUMAN_A, "human"),
            (AI_A, "artificer"),
            (AI_B, "artificer"),
        )
        participants = _make_participants([HUMAN_A, AI_A, AI_B])
        events = [_make_event(mentions=[AI_A])]

        result = resolve_routing(
            _make_params(
                entities,
                participants,
                events=events,
                config=OrchestratorConfig(default_respondent_mode="mentioned_only"),
            )
        )

        assert result.rule_applied == "explicit_mention"
        assert result.selected_entity == AI_A

    def test_returns_no_dispatch_in_mentioned_only_mode_when_no_ai_is_mentioned(
        self,
    ) -> None:
        entities = _make_entities(
            (HUMAN_A, "human"),
            (AI_A, "artificer"),
        )
        participants = _make_participants([HUMAN_A, AI_A])

        result = resolve_routing(
            _make_params(
                entities,
                participants,
                config=OrchestratorConfig(default_respondent_mode="mentioned_only"),
            )
        )

        assert result.rule_applied == "mentioned_only_mode"
        assert result.selected_entity is None

    def test_selects_the_only_ai_participant_in_auto_mode(self) -> None:
        entities = _make_entities(
            (HUMAN_A, "human"),
            (AI_A, "artificer"),
        )
        participants = _make_participants([HUMAN_A, AI_A])

        result = resolve_routing(
            _make_params(
                entities,
                participants,
                config=OrchestratorConfig(default_respondent_mode="auto"),
            )
        )

        assert result.selected_entity == AI_A
        assert result.eligible_entities == [AI_A]

    def test_prefers_domain_match_over_no_match_in_auto_mode(self) -> None:
        entities = _make_entities(
            (HUMAN_A, "human"),
            (AI_A, "artificer", ["code-review"]),
            (AI_B, "artificer", ["design"]),
        )
        participants = _make_participants([HUMAN_A, AI_A, AI_B])
        events = [_make_event(text="Please do a code-review of this PR")]

        result = resolve_routing(
            _make_params(
                entities,
                participants,
                events=events,
                config=OrchestratorConfig(default_respondent_mode="auto"),
            )
        )

        assert result.rule_applied == "auto_domain_match"
        assert result.selected_entity == AI_A

    def test_uses_round_robin_when_domain_scores_are_equal_in_auto_mode(
        self,
    ) -> None:
        entities = _make_entities(
            (HUMAN_A, "human"),
            (AI_A, "artificer"),
            (AI_B, "artificer"),
        )
        participants = _make_participants([HUMAN_A, AI_A, AI_B])

        # AI_A was served more recently
        last_served = {AI_A: 1000.0, AI_B: 500.0}

        result = resolve_routing(
            _make_params(
                entities,
                participants,
                config=OrchestratorConfig(default_respondent_mode="auto"),
                last_served=last_served,
            )
        )

        assert result.rule_applied == "auto_round_robin"
        assert result.selected_entity == AI_B  # B served less recently

    def test_excludes_batch_author_from_eligible_entities(self) -> None:
        entities = _make_entities(
            (HUMAN_A, "human"),
            (AI_A, "artificer"),
            (AI_B, "artificer"),
        )
        participants = _make_participants([HUMAN_A, AI_A, AI_B])

        # Batch authored by AI_A (cascade scenario)
        batch = _make_batch(author_id=AI_A)

        result = resolve_routing(
            _make_params(
                entities,
                participants,
                batch=batch,
                config=OrchestratorConfig(default_respondent_mode="auto"),
            )
        )

        assert result.selected_entity == AI_B
        assert AI_A not in result.eligible_entities

    def test_applies_handoff_override_regardless_of_mode(self) -> None:
        entities = _make_entities(
            (HUMAN_A, "human"),
            (AI_A, "artificer"),
            (AI_B, "artificer"),
        )
        participants = _make_participants([HUMAN_A, AI_A, AI_B])

        handoff = HandoffContent(
            from_entity=AI_A,
            to_entity=AI_B,
            reason="Needs design expertise",
            context_summary="Routing test",
            source_event=EventId("evt-1"),
        )

        result = resolve_routing(
            _make_params(
                entities,
                participants,
                config=OrchestratorConfig(default_respondent_mode="mentioned_only"),
                active_handoff=handoff,
            )
        )

        assert result.rule_applied == "handoff_override"
        assert result.selected_entity == AI_B

    def test_gives_first_mentioned_priority_when_multiple_ais_are_mentioned(
        self,
    ) -> None:
        entities = _make_entities(
            (HUMAN_A, "human"),
            (AI_A, "artificer"),
            (AI_B, "artificer"),
        )
        participants = _make_participants([HUMAN_A, AI_A, AI_B])
        events = [_make_event(mentions=[AI_B, AI_A])]

        result = resolve_routing(_make_params(entities, participants, events=events))

        assert result.rule_applied == "explicit_mention"
        assert result.selected_entity == AI_B  # first mentioned
        assert AI_A in result.eligible_entities
        assert AI_B in result.eligible_entities

    def test_filters_human_entities_from_eligible_list(self) -> None:
        entities = _make_entities(
            (HUMAN_A, "human"),
            (AI_A, "artificer"),
        )
        participants = _make_participants([HUMAN_A, AI_A])

        result = resolve_routing(
            _make_params(
                entities,
                participants,
                config=OrchestratorConfig(default_respondent_mode="auto"),
            )
        )

        assert HUMAN_A not in result.eligible_entities
        assert result.selected_entity == AI_A

    def test_prefers_artificer_over_daemon_in_auto_mode(self) -> None:
        entities = _make_entities(
            (HUMAN_A, "human"),
            (DAEMON_A, "daemon"),
            (AI_A, "artificer"),
        )
        participants = _make_participants([HUMAN_A, DAEMON_A, AI_A])

        result = resolve_routing(
            _make_params(
                entities,
                participants,
                config=OrchestratorConfig(default_respondent_mode="auto"),
            )
        )

        assert result.selected_entity == AI_A

    def test_explicit_mention_overrides_mentioned_only_fallback(self) -> None:
        entities = _make_entities(
            (HUMAN_A, "human"),
            (AI_A, "artificer"),
            (AI_B, "artificer"),
        )
        participants = _make_participants([HUMAN_A, AI_A, AI_B])
        events = [_make_event(mentions=[AI_B])]

        result = resolve_routing(
            _make_params(
                entities,
                participants,
                events=events,
                config=OrchestratorConfig(default_respondent_mode="mentioned_only"),
            )
        )

        assert result.rule_applied == "explicit_mention"
        assert result.selected_entity == AI_B
