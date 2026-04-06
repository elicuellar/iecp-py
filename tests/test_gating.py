"""Gating Engine tests -- Phase 4: Orchestration.

Pure function tests -- no mocks, no timers, fully deterministic.
"""

from __future__ import annotations

from iecp_core.orchestrator.gating import GatingParams, evaluate_gating
from iecp_core.orchestrator.types import DEFAULT_ORCHESTRATOR_CONFIG, OrchestratorConfig
from iecp_core.types.entity import EntityId, EntityLifecycleStatus
from iecp_core.types.event import ConversationId

# -- Helpers -----------------------------------------------------------------

ENTITY_A = EntityId("entity-a")
CONV_ID = ConversationId("conv-gating")


def _make_params(
    entity_id: EntityId = ENTITY_A,
    conversation_id: ConversationId = CONV_ID,
    ai_depth_counter: int = 0,
    config: OrchestratorConfig | None = None,
    hourly_invocation_count: int = 0,
    concurrent_processing_count: int = 0,
    entity_status: EntityLifecycleStatus = "active",
    escalation_active: bool = False,
) -> GatingParams:
    return GatingParams(
        entity_id=entity_id,
        conversation_id=conversation_id,
        ai_depth_counter=ai_depth_counter,
        config=config or OrchestratorConfig(**DEFAULT_ORCHESTRATOR_CONFIG.__dict__),
        hourly_invocation_count=hourly_invocation_count,
        concurrent_processing_count=concurrent_processing_count,
        entity_status=entity_status,
        escalation_active=escalation_active,
    )


# -- Tests -------------------------------------------------------------------


class TestEvaluateGating:
    def test_allows_dispatch_when_all_checks_pass(self) -> None:
        result = evaluate_gating(_make_params())

        assert result.allowed is True
        assert result.reason is None
        assert len(result.checks) == 5
        assert all(c.passed for c in result.checks)

    def test_blocks_when_cascade_depth_is_exceeded(self) -> None:
        result = evaluate_gating(
            _make_params(
                ai_depth_counter=3,
                config=OrchestratorConfig(max_cascade_depth=3),
            )
        )

        assert result.allowed is False
        assert "Depth 3 >= max 3" in (result.reason or "")

        cascade_check = next(
            (c for c in result.checks if c.name == "cascade_depth"), None
        )
        assert cascade_check is not None
        assert cascade_check.passed is False

    def test_blocks_when_rate_limit_is_exceeded(self) -> None:
        result = evaluate_gating(
            _make_params(
                hourly_invocation_count=60,
                config=OrchestratorConfig(max_ai_invocations_per_hour=60),
            )
        )

        assert result.allowed is False
        assert "60 invocations >= max 60/hr" in (result.reason or "")

        rate_check = next(
            (c for c in result.checks if c.name == "rate_limit"), None
        )
        assert rate_check is not None
        assert rate_check.passed is False

    def test_blocks_when_concurrency_limit_is_exceeded(self) -> None:
        result = evaluate_gating(
            _make_params(
                concurrent_processing_count=1,
                config=OrchestratorConfig(max_concurrent_ai_processing=1),
            )
        )

        assert result.allowed is False

        concurrency_check = next(
            (c for c in result.checks if c.name == "concurrency"), None
        )
        assert concurrency_check is not None
        assert concurrency_check.passed is False

    def test_blocks_when_entity_is_disconnected(self) -> None:
        result = evaluate_gating(_make_params(entity_status="disconnected"))

        assert result.allowed is False

        status_check = next(
            (c for c in result.checks if c.name == "entity_status"), None
        )
        assert status_check is not None
        assert status_check.passed is False
        assert "disconnected" in (status_check.detail or "")

    def test_blocks_when_entity_has_left(self) -> None:
        result = evaluate_gating(_make_params(entity_status="left"))

        assert result.allowed is False

        status_check = next(
            (c for c in result.checks if c.name == "entity_status"), None
        )
        assert status_check is not None
        assert status_check.passed is False

    def test_blocks_when_entity_is_already_processing(self) -> None:
        result = evaluate_gating(_make_params(entity_status="processing"))

        assert result.allowed is False

        status_check = next(
            (c for c in result.checks if c.name == "entity_status"), None
        )
        assert status_check is not None
        assert status_check.passed is False

    def test_blocks_when_escalation_is_active(self) -> None:
        result = evaluate_gating(_make_params(escalation_active=True))

        assert result.allowed is False

        escalation_check = next(
            (c for c in result.checks if c.name == "escalation"), None
        )
        assert escalation_check is not None
        assert escalation_check.passed is False
        assert "Escalation active" in (escalation_check.detail or "")

    def test_reports_all_failures_when_multiple_checks_fail(self) -> None:
        result = evaluate_gating(
            _make_params(
                ai_depth_counter=5,
                hourly_invocation_count=100,
                escalation_active=True,
                entity_status="disconnected",
            )
        )

        assert result.allowed is False

        failed_checks = [c for c in result.checks if not c.passed]
        assert len(failed_checks) >= 3
        assert result.reason is not None

    def test_allows_entity_with_idle_status(self) -> None:
        result = evaluate_gating(_make_params(entity_status="idle"))

        assert result.allowed is True
        status_check = next(
            (c for c in result.checks if c.name == "entity_status"), None
        )
        assert status_check is not None
        assert status_check.passed is True

    def test_allows_when_just_below_all_limits(self) -> None:
        result = evaluate_gating(
            _make_params(
                ai_depth_counter=2,
                hourly_invocation_count=59,
                concurrent_processing_count=0,
                config=OrchestratorConfig(
                    max_cascade_depth=3, max_ai_invocations_per_hour=60
                ),
            )
        )

        assert result.allowed is True
        assert all(c.passed for c in result.checks)
