"""Attention Signals -- Phase 9: Signals (SS18 of the IECP specification)."""

from .attention_signal_manager import AttentionSignalManager
from .types import (
    DEFAULT_ATTENTION_SIGNAL_CONFIG,
    ActiveSignal,
    AttentionSignalConfig,
    AttentionSignalType,
)

__all__ = [
    "AttentionSignalManager",
    "ActiveSignal",
    "AttentionSignalConfig",
    "AttentionSignalType",
    "DEFAULT_ATTENTION_SIGNAL_CONFIG",
]
