"""Gating Engine -- Pure function for AI dispatch gating.

Evaluates whether a selected AI entity is allowed to be dispatched.
All checks must pass for dispatch to proceed. No side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types.entity import EntityId, EntityLifecycleStatus
from ..types.event import ConversationId
from .types import GatingCheck, GatingResult, OrchestratorConfig


@dataclass
class GatingParams:
    """Input parameters for the gating function."""

    entity_id: EntityId
    conversation_id: ConversationId
    ai_depth_counter: int
    config: OrchestratorConfig
    hourly_invocation_count: int
    concurrent_processing_count: int
    entity_status: EntityLifecycleStatus
    escalation_active: bool


def evaluate_gating(params: GatingParams) -> GatingResult:
    """Evaluate whether an AI entity should be dispatched.

    All checks must pass:
    1. Cascade depth < max
    2. Rate limit not exceeded
    3. Concurrent processing limit not exceeded
    4. Entity is active, idle, or joined
    5. No active escalation
    """
    checks: list[GatingCheck] = []

    # 1. Cascade depth
    cascade_pass = params.ai_depth_counter < params.config.max_cascade_depth
    checks.append(
        GatingCheck(
            name="cascade_depth",
            passed=cascade_pass,
            detail=(
                f"Depth {params.ai_depth_counter} < max {params.config.max_cascade_depth}"
                if cascade_pass
                else f"Depth {params.ai_depth_counter} >= max {params.config.max_cascade_depth}"
            ),
        )
    )

    # 2. Rate limit
    rate_pass = (
        params.hourly_invocation_count < params.config.max_ai_invocations_per_hour
    )
    checks.append(
        GatingCheck(
            name="rate_limit",
            passed=rate_pass,
            detail=(
                f"{params.hourly_invocation_count} invocations < max {params.config.max_ai_invocations_per_hour}/hr"
                if rate_pass
                else f"{params.hourly_invocation_count} invocations >= max {params.config.max_ai_invocations_per_hour}/hr"
            ),
        )
    )

    # 3. Concurrency
    concurrency_pass = (
        params.concurrent_processing_count
        < params.config.max_concurrent_ai_processing
    )
    checks.append(
        GatingCheck(
            name="concurrency",
            passed=concurrency_pass,
            detail=(
                f"{params.concurrent_processing_count} processing < max {params.config.max_concurrent_ai_processing}"
                if concurrency_pass
                else f"{params.concurrent_processing_count} processing >= max {params.config.max_concurrent_ai_processing}"
            ),
        )
    )

    # 4. Entity status
    valid_statuses: list[EntityLifecycleStatus] = ["joined", "active", "idle"]
    status_pass = params.entity_status in valid_statuses
    checks.append(
        GatingCheck(
            name="entity_status",
            passed=status_pass,
            detail=(
                f"Entity status '{params.entity_status}' is dispatchable"
                if status_pass
                else f"Entity status '{params.entity_status}' is not dispatchable (must be active or idle)"
            ),
        )
    )

    # 5. Escalation
    escalation_pass = not params.escalation_active
    checks.append(
        GatingCheck(
            name="escalation",
            passed=escalation_pass,
            detail=(
                "No active escalation"
                if escalation_pass
                else "Escalation active — AI dispatch suppressed until human responds"
            ),
        )
    )

    all_passed = all(c.passed for c in checks)
    failed_checks = [c for c in checks if not c.passed]

    return GatingResult(
        allowed=all_passed,
        reason=(
            None
            if all_passed
            else "; ".join(c.detail or c.name for c in failed_checks)
        ),
        checks=checks,
    )
