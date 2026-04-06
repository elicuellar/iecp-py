"""Debounce -- Phase 2: Smart Debouncing & Batch Sealing."""

from .debouncer import Debouncer
from .types import (
    DEFAULT_DEBOUNCER_CONFIG,
    DebouncerConfig,
    DefaultTimerProvider,
    SealedBatch,
    TimerProvider,
    default_timer_provider,
)

__all__ = [
    "DEFAULT_DEBOUNCER_CONFIG",
    "Debouncer",
    "DebouncerConfig",
    "DefaultTimerProvider",
    "SealedBatch",
    "TimerProvider",
    "default_timer_provider",
]
