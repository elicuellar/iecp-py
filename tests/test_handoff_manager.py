"""Tests for HandoffManager -- SS20 of the specification."""

from unittest.mock import patch

import pytest

from iecp_core.handoff import HandoffManager, HandoffManagerConfig
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId, EventId

AI_A = EntityId("ai-a")
AI_B = EntityId("ai-b")
AI_C = EntityId("ai-c")
AI_D = EntityId("ai-d")
HUMAN = EntityId("human-1")
CONV_1 = ConversationId("conv-1")
EVT_1 = EventId("evt-1")
EVT_2 = EventId("evt-2")
EVT_3 = EventId("evt-3")
EVT_4 = EventId("evt-4")
SRC_1 = EventId("src-1")


class TestHandoffManager:
    def setup_method(self) -> None:
        self.manager = HandoffManager()

    def teardown_method(self) -> None:
        self.manager.destroy()

    # -- Handoff Tests --------------------------------------------------------

    def test_register_handoff_successfully(self) -> None:
        result = self.manager.handoff(
            event_id=EVT_1,
            conversation_id=CONV_1,
            from_entity=AI_A,
            to_entity=AI_B,
            reason="Architecture question outside my domain",
            context_summary="User asked about database design",
            source_event=SRC_1,
        )

        assert result["success"] is True
        assert result["handoff"] is not None
        assert result["handoff"].from_entity == AI_A
        assert result["handoff"].to_entity == AI_B
        assert result["handoff"].chain_depth == 1

    def test_track_handoff_chain(self) -> None:
        first = self.manager.handoff(
            event_id=EVT_1,
            conversation_id=CONV_1,
            from_entity=AI_A,
            to_entity=AI_B,
            reason="Passing to B",
            context_summary="Context",
            source_event=SRC_1,
        )
        assert first["success"] is True
        assert first["handoff"].chain_depth == 1

        second = self.manager.handoff(
            event_id=EVT_2,
            conversation_id=CONV_1,
            from_entity=AI_B,
            to_entity=AI_C,
            reason="Passing to C",
            context_summary="More context",
            source_event=SRC_1,
        )
        assert second["success"] is True
        assert second["handoff"].chain_depth == 2
        assert self.manager.get_chain_depth(CONV_1) == 2

    def test_enforce_chain_depth_limit(self) -> None:
        # Depth 1: A->B
        self.manager.handoff(
            event_id=EVT_1,
            conversation_id=CONV_1,
            from_entity=AI_A,
            to_entity=AI_B,
            reason="To B",
            context_summary="ctx",
            source_event=SRC_1,
        )

        # Depth 2: B->C
        self.manager.handoff(
            event_id=EVT_2,
            conversation_id=CONV_1,
            from_entity=AI_B,
            to_entity=AI_C,
            reason="To C",
            context_summary="ctx",
            source_event=SRC_1,
        )

        # Depth 3: C->D
        self.manager.handoff(
            event_id=EVT_3,
            conversation_id=CONV_1,
            from_entity=AI_C,
            to_entity=AI_D,
            reason="To D",
            context_summary="ctx",
            source_event=SRC_1,
        )

        # Depth 4: D->A -- should be rejected (max_chain_depth = 3)
        result = self.manager.handoff(
            event_id=EVT_4,
            conversation_id=CONV_1,
            from_entity=AI_D,
            to_entity=AI_A,
            reason="Back to A",
            context_summary="ctx",
            source_event=SRC_1,
        )

        assert result["success"] is False
        assert "chain depth limit exceeded" in result["error"]

    def test_expire_handoff_after_expiry_ms(self) -> None:
        manager = HandoffManager(
            HandoffManagerConfig(max_chain_depth=3, handoff_expiry_ms=1000)
        )
        try:
            fake_now = 1000.0
            with patch("iecp_core.handoff.handoff_manager.time") as mock_time:
                mock_time.time.return_value = fake_now
                manager.handoff(
                    event_id=EVT_1,
                    conversation_id=CONV_1,
                    from_entity=AI_A,
                    to_entity=AI_B,
                    reason="Expiring",
                    context_summary="ctx",
                    source_event=SRC_1,
                )

                assert manager.get_active_handoff(CONV_1) is not None

                mock_time.time.return_value = fake_now + 1.001
                assert manager.get_active_handoff(CONV_1) is None
        finally:
            manager.destroy()

    def test_resolve_handoff_clears_and_resets(self) -> None:
        self.manager.handoff(
            event_id=EVT_1,
            conversation_id=CONV_1,
            from_entity=AI_A,
            to_entity=AI_B,
            reason="Handoff",
            context_summary="ctx",
            source_event=SRC_1,
        )

        assert self.manager.get_chain_depth(CONV_1) == 1
        assert self.manager.get_active_handoff(CONV_1) is not None

        self.manager.resolve_handoff(CONV_1)

        assert self.manager.get_active_handoff(CONV_1) is None
        assert self.manager.get_chain_depth(CONV_1) == 0

    def test_track_chain_depth_correctly(self) -> None:
        assert self.manager.get_chain_depth(CONV_1) == 0

        self.manager.handoff(
            event_id=EVT_1,
            conversation_id=CONV_1,
            from_entity=AI_A,
            to_entity=AI_B,
            reason="r",
            context_summary="c",
            source_event=SRC_1,
        )
        assert self.manager.get_chain_depth(CONV_1) == 1

        self.manager.handoff(
            event_id=EVT_2,
            conversation_id=CONV_1,
            from_entity=AI_B,
            to_entity=AI_C,
            reason="r",
            context_summary="c",
            source_event=SRC_1,
        )
        assert self.manager.get_chain_depth(CONV_1) == 2

    # -- Escalation Tests -----------------------------------------------------

    def test_register_escalation_successfully(self) -> None:
        esc = self.manager.escalate(
            event_id=EVT_1,
            conversation_id=CONV_1,
            entity_id=AI_A,
            reason="Need human approval for deployment",
            requires="approval",
            context_summary="Production deploy ready",
            source_event=SRC_1,
        )

        assert esc.resolved is False
        assert esc.requires == "approval"
        assert esc.entity_id == AI_A

    def test_is_escalation_active_true(self) -> None:
        self.manager.escalate(
            event_id=EVT_1,
            conversation_id=CONV_1,
            entity_id=AI_A,
            reason="Need approval",
            requires="approval",
            context_summary="ctx",
            source_event=SRC_1,
        )

        assert self.manager.is_escalation_active(CONV_1) is True

    def test_resolve_escalation_by_human(self) -> None:
        self.manager.escalate(
            event_id=EVT_1,
            conversation_id=CONV_1,
            entity_id=AI_A,
            reason="Need approval",
            requires="approval",
            context_summary="ctx",
            source_event=SRC_1,
        )

        assert self.manager.is_escalation_active(CONV_1) is True

        self.manager.resolve_escalation(CONV_1, HUMAN)

        assert self.manager.is_escalation_active(CONV_1) is False

        esc = self.manager.get_active_escalation(CONV_1)
        assert esc is None

    def test_keep_only_latest_escalation(self) -> None:
        self.manager.escalate(
            event_id=EVT_1,
            conversation_id=CONV_1,
            entity_id=AI_A,
            reason="First escalation",
            requires="approval",
            context_summary="ctx",
            source_event=SRC_1,
        )

        self.manager.escalate(
            event_id=EVT_2,
            conversation_id=CONV_1,
            entity_id=AI_B,
            reason="Second escalation",
            requires="decision",
            context_summary="ctx2",
            source_event=SRC_1,
        )

        active = self.manager.get_active_escalation(CONV_1)
        assert active is not None
        assert active.event_id == EVT_2
        assert active.reason == "Second escalation"

    def test_clean_up_on_destroy(self) -> None:
        self.manager.handoff(
            event_id=EVT_1,
            conversation_id=CONV_1,
            from_entity=AI_A,
            to_entity=AI_B,
            reason="r",
            context_summary="c",
            source_event=SRC_1,
        )
        self.manager.escalate(
            event_id=EVT_2,
            conversation_id=CONV_1,
            entity_id=AI_A,
            reason="r",
            requires="approval",
            context_summary="c",
            source_event=SRC_1,
        )

        self.manager.destroy()

        assert self.manager.get_active_handoff(CONV_1) is None
        assert self.manager.is_escalation_active(CONV_1) is False
        assert self.manager.get_chain_depth(CONV_1) == 0
