"""Conversation Summarizer -- Phase 5 (section 9.3) of the IECP specification.

Defines the interface for conversation summarization and provides
a simple V1 implementation that concatenates bullet points.
"""

from __future__ import annotations

from typing import Protocol

from ..types.event import ConversationId, Event, MessageContent
from .token_estimator import SimpleTokenEstimator
from .types import TokenEstimator


class ConversationSummarizer(Protocol):
    """Interface for conversation summarization."""

    async def summarize(
        self,
        *,
        conversation_id: ConversationId,
        events: list[Event],
        existing_summary: str | None,
    ) -> str: ...


class SimpleSummarizer:
    """Simple V1 summarizer that concatenates events as bullet points.

    Prepends existing summary, then adds new events.
    Keeps the total under the token budget by truncating older content.
    """

    def __init__(
        self,
        *,
        estimator: TokenEstimator | None = None,
        max_tokens: int = 1_000,
    ) -> None:
        self._estimator: TokenEstimator = estimator or SimpleTokenEstimator()
        self._max_tokens = max_tokens

    async def summarize(
        self,
        *,
        conversation_id: ConversationId,
        events: list[Event],
        existing_summary: str | None,
    ) -> str:
        # Build bullet points from events
        bullets: list[str] = []
        for e in events:
            if e.type not in ("message", "action", "decision"):
                continue

            if e.type == "message":
                assert isinstance(e.content, MessageContent)
                text = e.content.text
                if len(text) > 100:
                    text = text[:100] + "..."
                bullets.append(f"- [{e.author_id}]: {text}")
            elif e.type == "decision":
                # DecisionContent has summary
                content = e.content
                bullets.append(f"- [decision]: {content.summary}")  # type: ignore[union-attr]
            else:
                # ActionContent has description
                content = e.content
                bullets.append(f"- [action]: {content.description}")  # type: ignore[union-attr]

        new_content = "\n".join(bullets)

        # Combine with existing summary
        if existing_summary:
            result = f"{existing_summary}\n{new_content}"
        else:
            result = new_content

        # Truncate from the front if over budget
        tokens = self._estimator.estimate(result)
        if tokens > self._max_tokens:
            max_chars = self._max_tokens * 4  # reverse the 4 chars/token heuristic
            result = "..." + result[len(result) - max_chars + 3 :]

        return result
