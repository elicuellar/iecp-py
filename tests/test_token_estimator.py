"""Token estimator tests -- Phase 5: Context Assembly."""

from __future__ import annotations

import json
import math

from iecp_core.context.token_estimator import SimpleTokenEstimator
from iecp_core.events.event_factory import create_message_event
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId


def _create_test_event(
    text: str = "Hello",
    metadata: dict | None = None,
) -> "iecp_core.types.event.Event":
    import iecp_core.types.event as evt_mod

    return create_message_event(
        conversation_id=ConversationId("conv-test"),
        author_id=EntityId("author-test"),
        author_type="human",
        text=text,
        metadata=metadata,
    )


class TestSimpleTokenEstimator:
    def setup_method(self) -> None:
        self.estimator = SimpleTokenEstimator()

    def test_returns_0_for_empty_string(self) -> None:
        assert self.estimator.estimate("") == 0

    def test_estimates_1_token_per_4_characters(self) -> None:
        # 12 chars -> ceil(12/4) = 3 tokens
        assert self.estimator.estimate("Hello World!") == 3

    def test_rounds_up_partial_tokens(self) -> None:
        # 5 chars -> ceil(5/4) = 2 tokens
        assert self.estimator.estimate("Hello") == 2

    def test_handles_single_character(self) -> None:
        # 1 char -> ceil(1/4) = 1 token
        assert self.estimator.estimate("a") == 1

    def test_estimates_longer_string_approximately_correctly(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."  # 44 chars
        tokens = self.estimator.estimate(text)
        assert tokens == 11  # ceil(44/4)

    def test_estimates_event_tokens_by_serializing_to_json(self) -> None:
        event = _create_test_event(text="Hello world")
        tokens = self.estimator.estimate_event(event)
        json_length = len(json.dumps(event.model_dump(), default=str))
        assert tokens == math.ceil(json_length / 4)
        assert tokens > 0

    def test_event_estimation_includes_all_fields(self) -> None:
        event = _create_test_event(
            text="Test message with metadata",
            metadata={"key": "value", "nested": {"deep": True}},
        )
        tokens = self.estimator.estimate_event(event)
        # Should be larger than or equal to a minimal event because of metadata
        minimal_event = _create_test_event(text="Test message with metadata")
        minimal_tokens = self.estimator.estimate_event(minimal_event)
        assert tokens >= minimal_tokens
