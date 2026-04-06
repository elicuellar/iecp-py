"""Tests for DecisionManager -- SS19 of the specification."""

import pytest

from iecp_core.decisions import DecisionManager, DecisionManagerConfig
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId, EventId

HUMAN = EntityId("human-1")
AI_1 = EntityId("ai-1")
AI_2 = EntityId("ai-2")
CONV_1 = ConversationId("conv-1")
CONV_2 = ConversationId("conv-2")
EVT_1 = EventId("evt-1")
EVT_2 = EventId("evt-2")
EVT_3 = EventId("evt-3")
EVT_4 = EventId("evt-4")
CTX_1 = EventId("ctx-1")
CTX_2 = EventId("ctx-2")


class TestDecisionManager:
    def setup_method(self) -> None:
        self.manager = DecisionManager()

    def test_create_decision_in_proposed_state(self) -> None:
        decision = self.manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Use WebGL for rendering",
            proposed_by=AI_1,
            context_events=[CTX_1],
        )

        assert decision.status == "proposed"
        assert decision.event_id == EVT_1
        assert decision.summary == "Use WebGL for rendering"
        assert decision.proposed_by == AI_1
        assert decision.affirmed_by == []
        assert decision.rejected_by == []
        assert decision.context_events == [CTX_1]
        assert decision.superseded_by is None

    def test_affirm_decision(self) -> None:
        self.manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Use WebGL",
            proposed_by=AI_1,
            context_events=[],
        )

        affirmed = self.manager.affirm(EVT_1, HUMAN, True)
        assert affirmed is not None
        assert affirmed.status == "affirmed"
        assert HUMAN in affirmed.affirmed_by

    def test_reject_decision(self) -> None:
        self.manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Use Canvas 2D",
            proposed_by=AI_1,
            context_events=[],
        )

        rejected = self.manager.reject(EVT_1, HUMAN)
        assert rejected is not None
        assert rejected.status == "rejected"
        assert HUMAN in rejected.rejected_by

    def test_supersede_decision(self) -> None:
        self.manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Use Canvas 2D",
            proposed_by=AI_1,
            context_events=[CTX_1],
        )

        result = self.manager.supersede(
            EVT_1,
            event_id=EVT_2,
            conversation_id=CONV_1,
            summary="Use WebGL instead",
            proposed_by=AI_2,
            context_events=[CTX_1, CTX_2],
        )

        assert result is not None
        assert result["old"].status == "superseded"
        assert result["old"].superseded_by == EVT_2
        assert result["new"].status == "proposed"
        assert result["new"].summary == "Use WebGL instead"

    def test_enforce_human_affirmation_ai_cannot_affirm(self) -> None:
        self.manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Architecture decision",
            proposed_by=AI_1,
            context_events=[],
        )

        # AI tries to affirm (is_human=False)
        result = self.manager.affirm(EVT_1, AI_2, False)
        assert result is None

        # Decision should still be proposed
        decision = self.manager.get_decision(EVT_1)
        assert decision is not None
        assert decision.status == "proposed"

    def test_allow_ai_affirm_when_not_required(self) -> None:
        manager = DecisionManager(
            DecisionManagerConfig(require_human_affirmation=False)
        )

        manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Minor refactor",
            proposed_by=AI_1,
            context_events=[],
        )

        result = manager.affirm(EVT_1, AI_2, False)
        assert result is not None
        assert result.status == "affirmed"

    def test_active_decisions_exclude_rejected_superseded(self) -> None:
        # Proposed
        self.manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Decision A",
            proposed_by=AI_1,
            context_events=[],
        )

        # Affirmed
        self.manager.propose(
            event_id=EVT_2,
            conversation_id=CONV_1,
            summary="Decision B",
            proposed_by=AI_1,
            context_events=[],
        )
        self.manager.affirm(EVT_2, HUMAN, True)

        # Rejected
        self.manager.propose(
            event_id=EVT_3,
            conversation_id=CONV_1,
            summary="Decision C",
            proposed_by=AI_1,
            context_events=[],
        )
        self.manager.reject(EVT_3, HUMAN)

        # Superseded
        self.manager.propose(
            event_id=EVT_4,
            conversation_id=CONV_1,
            summary="Decision D",
            proposed_by=AI_1,
            context_events=[],
        )
        self.manager.supersede(
            EVT_4,
            event_id=EventId("evt-5"),
            conversation_id=CONV_1,
            summary="Decision D v2",
            proposed_by=AI_1,
            context_events=[],
        )

        active = self.manager.get_active_decisions(CONV_1)
        # EVT_1 (proposed), EVT_2 (affirmed), evt-5 (proposed from supersede)
        assert len(active) == 3
        summaries = sorted(d.summary for d in active)
        assert summaries == ["Decision A", "Decision B", "Decision D v2"]

    def test_multiple_decisions_per_conversation(self) -> None:
        self.manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Decision 1",
            proposed_by=AI_1,
            context_events=[],
        )
        self.manager.propose(
            event_id=EVT_2,
            conversation_id=CONV_1,
            summary="Decision 2",
            proposed_by=AI_2,
            context_events=[],
        )
        self.manager.propose(
            event_id=EVT_3,
            conversation_id=CONV_2,
            summary="Decision 3",
            proposed_by=AI_1,
            context_events=[],
        )

        assert len(self.manager.get_all_decisions(CONV_1)) == 2
        assert len(self.manager.get_all_decisions(CONV_2)) == 1

    def test_track_multiple_affirmers(self) -> None:
        self.manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Architecture choice",
            proposed_by=AI_1,
            context_events=[],
        )

        self.manager.affirm(EVT_1, HUMAN, True)
        self.manager.affirm(EVT_1, EntityId("human-2"), True)

        decision = self.manager.get_decision(EVT_1)
        assert decision is not None
        assert len(decision.affirmed_by) == 2
        assert HUMAN in decision.affirmed_by
        assert EntityId("human-2") in decision.affirmed_by

    def test_no_duplicate_affirmers(self) -> None:
        self.manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Choice",
            proposed_by=AI_1,
            context_events=[],
        )

        self.manager.affirm(EVT_1, HUMAN, True)
        self.manager.affirm(EVT_1, HUMAN, True)  # duplicate

        decision = self.manager.get_decision(EVT_1)
        assert decision is not None
        assert len(decision.affirmed_by) == 1

    def test_return_none_affirming_nonexistent(self) -> None:
        result = self.manager.affirm(EventId("nonexistent"), HUMAN, True)
        assert result is None

    def test_reject_already_affirmed(self) -> None:
        self.manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Decision",
            proposed_by=AI_1,
            context_events=[],
        )

        self.manager.affirm(EVT_1, HUMAN, True)
        assert self.manager.get_decision(EVT_1).status == "affirmed"

        self.manager.reject(EVT_1, EntityId("human-2"))
        assert self.manager.get_decision(EVT_1).status == "rejected"

    def test_return_none_superseding_nonexistent(self) -> None:
        result = self.manager.supersede(
            EventId("nonexistent"),
            event_id=EVT_2,
            conversation_id=CONV_1,
            summary="New",
            proposed_by=AI_1,
            context_events=[],
        )
        assert result is None

    def test_get_decision_specific(self) -> None:
        self.manager.propose(
            event_id=EVT_1,
            conversation_id=CONV_1,
            summary="Specific decision",
            proposed_by=AI_1,
            context_events=[CTX_1],
        )

        decision = self.manager.get_decision(EVT_1)
        assert decision is not None
        assert decision.summary == "Specific decision"

        # Non-existent
        assert self.manager.get_decision(EventId("nope")) is None
