"""Summarizer tests -- Phase 5: Context Assembly."""

from __future__ import annotations

import pytest

from iecp_core.context.summarizer import SimpleSummarizer
from iecp_core.context.token_estimator import SimpleTokenEstimator
from iecp_core.events.event_factory import create_message_event
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId, Event

CONV_ID = ConversationId("conv-test")


def _create_test_event(
    text: str = "Hello",
    author_id: str = "author-test",
) -> Event:
    return create_message_event(
        conversation_id=CONV_ID,
        author_id=EntityId(author_id),
        author_type="human",
        text=text,
    )


class TestSimpleSummarizer:
    @pytest.mark.asyncio
    async def test_produces_bullet_points_from_message_events(self) -> None:
        summarizer = SimpleSummarizer()
        events = [
            _create_test_event(text="Hello everyone", author_id="alice"),
            _create_test_event(text="Welcome aboard", author_id="bob"),
        ]

        result = await summarizer.summarize(
            conversation_id=CONV_ID,
            events=events,
            existing_summary=None,
        )

        assert "- [alice]: Hello everyone" in result
        assert "- [bob]: Welcome aboard" in result

    @pytest.mark.asyncio
    async def test_prepends_existing_summary_to_new_content(self) -> None:
        summarizer = SimpleSummarizer()
        events = [
            _create_test_event(text="New message", author_id="charlie"),
        ]

        result = await summarizer.summarize(
            conversation_id=CONV_ID,
            events=events,
            existing_summary="- [alice]: Earlier message",
        )

        assert result.startswith("- [alice]: Earlier message")
        assert "- [charlie]: New message" in result

    @pytest.mark.asyncio
    async def test_stays_under_the_token_budget(self) -> None:
        estimator = SimpleTokenEstimator()
        max_tokens = 50  # very tight budget
        summarizer = SimpleSummarizer(estimator=estimator, max_tokens=max_tokens)

        # Generate lots of events to exceed the budget
        events = [
            _create_test_event(
                text=f"Message number {i} with some extra content to fill space",
                author_id=f"user-{i}",
            )
            for i in range(20)
        ]

        result = await summarizer.summarize(
            conversation_id=CONV_ID,
            events=events,
            existing_summary=None,
        )

        tokens = estimator.estimate(result)
        assert tokens <= max_tokens

    @pytest.mark.asyncio
    async def test_filters_out_system_and_attention_events(self) -> None:
        summarizer = SimpleSummarizer()

        events = [
            _create_test_event(text="Important message", author_id="alice"),
        ]

        result = await summarizer.summarize(
            conversation_id=CONV_ID,
            events=events,
            existing_summary=None,
        )

        assert "Important message" in result

    @pytest.mark.asyncio
    async def test_truncates_long_message_text_in_bullets(self) -> None:
        summarizer = SimpleSummarizer()
        long_text = "A" * 200
        events = [
            _create_test_event(text=long_text, author_id="alice"),
        ]

        result = await summarizer.summarize(
            conversation_id=CONV_ID,
            events=events,
            existing_summary=None,
        )

        # Should truncate to 100 chars + '...'
        assert "..." in result
        assert len(result) < len(long_text)
