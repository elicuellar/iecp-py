"""Handoff Manager -- Phase 9: Handoffs (SS20 of the IECP specification)."""

from .handoff_manager import HandoffManager
from .types import (
    DEFAULT_HANDOFF_MANAGER_CONFIG,
    ActiveEscalation,
    ActiveHandoff,
    EscalationRequires,
    HandoffManagerConfig,
)

__all__ = [
    "HandoffManager",
    "ActiveEscalation",
    "ActiveHandoff",
    "EscalationRequires",
    "HandoffManagerConfig",
    "DEFAULT_HANDOFF_MANAGER_CONFIG",
]
