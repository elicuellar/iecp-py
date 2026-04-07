"""Orchestrator -- Phase 4: AI Dispatch Orchestration."""

from .gating import GatingParams, evaluate_gating
from .orchestrator import Orchestrator, OrchestratorConversationManager, OrchestratorEntityManager, OrchestratorEventStore
from .routing import RoutingParams, resolve_routing
from .types import (
    DEFAULT_ORCHESTRATOR_CONFIG,
    DispatchPayload,
    GatingCheck,
    GatingResult,
    OrchestratorConfig,
    OrchestratorError,
    OrchestrationTrace,
    RoutingDecision,
    RoutingRule,
)

__all__ = [
    "DEFAULT_ORCHESTRATOR_CONFIG",
    "DispatchPayload",
    "GatingCheck",
    "GatingParams",
    "GatingResult",
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorError",
    "OrchestrationTrace",
    "RoutingDecision",
    "RoutingParams",
    "RoutingRule",
    "OrchestratorConversationManager",
    "OrchestratorEntityManager",
    "OrchestratorEventStore",
    "evaluate_gating",
    "resolve_routing",
]
