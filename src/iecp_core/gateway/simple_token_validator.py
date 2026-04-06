"""Simple Token Validator -- Phase 7 of the IECP protocol.

In-memory token store for V1. Swappable for JWT, API key lookup, etc.
"""

from __future__ import annotations

from .types import AuthToken


class SimpleTokenValidator:
    """In-memory token validator."""

    def __init__(self, tokens: dict[str, AuthToken] | None = None) -> None:
        self._tokens: dict[str, AuthToken] = dict(tokens) if tokens else {}

    def add_token(self, token: str, auth: AuthToken) -> None:
        """Register a token."""
        self._tokens[token] = auth

    def remove_token(self, token: str) -> None:
        """Revoke a token."""
        self._tokens.pop(token, None)

    async def validate(self, token: str) -> AuthToken | None:
        """Validate a token. Returns the AuthToken if valid, None otherwise."""
        return self._tokens.get(token)
