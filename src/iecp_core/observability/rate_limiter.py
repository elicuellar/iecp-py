"""Rate Limiter -- §15.3 of the IECP specification.

Per-entity sliding window rate limiter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..types.entity import EntityId


# --- Configuration ----------------------------------------------------------


@dataclass(frozen=True)
class RateLimiterConfig:
    default_limit: int = 60
    """Default requests allowed per window."""

    window_ms: float = 60_000.0
    """Window size in milliseconds."""


DEFAULT_RATE_LIMITER_CONFIG = RateLimiterConfig()


# --- Check Result -----------------------------------------------------------


@dataclass
class RateLimitCheck:
    allowed: bool
    remaining: int
    reset_at: float


# --- RateLimiter ------------------------------------------------------------


class RateLimiter:
    """Per-entity sliding window rate limiter."""

    def __init__(self, config: RateLimiterConfig | dict | None = None) -> None:
        if config is None:
            self._config = DEFAULT_RATE_LIMITER_CONFIG
        elif isinstance(config, dict):
            self._config = RateLimiterConfig(**config)
        else:
            self._config = config

        self._custom_limits: dict[str, int] = {}
        self._requests: dict[str, list[float]] = {}

    def set_limit(self, entity_id: EntityId, limit: int) -> None:
        """Set a custom limit for an entity."""
        self._custom_limits[str(entity_id)] = limit

    def check(self, entity_id: EntityId) -> RateLimitCheck:
        """Check if an entity can make a request."""
        key = str(entity_id)
        limit = self._custom_limits.get(key, self._config.default_limit)
        now = time.time() * 1000
        window_start = now - self._config.window_ms

        timestamps = self._requests.get(key, [])
        in_window = [t for t in timestamps if t > window_start]

        # Prune old entries
        self._requests[key] = in_window

        remaining = max(0, limit - len(in_window))
        reset_at = (
            in_window[0] + self._config.window_ms if in_window else now + self._config.window_ms
        )

        return RateLimitCheck(
            allowed=len(in_window) < limit,
            remaining=remaining,
            reset_at=reset_at,
        )

    def record(self, entity_id: EntityId) -> None:
        """Record a request for an entity."""
        key = str(entity_id)
        timestamps = self._requests.get(key, [])
        timestamps.append(time.time() * 1000)
        self._requests[key] = timestamps

    def reset(self) -> None:
        """Reset all state."""
        self._custom_limits.clear()
        self._requests.clear()
