"""API Error Classes -- Phase 10 of the IECP protocol."""

from __future__ import annotations


class NotFoundError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ValidationError(Exception):
    def __init__(self, message: str, details: object = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class ConflictError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
