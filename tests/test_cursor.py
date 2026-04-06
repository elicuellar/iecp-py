from iecp_core.types.cursor import EntityCursor, has_unprocessed_events, is_cursor_order_valid
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId, EventId
from iecp_core.utils import generate_id


def _cursor(received: str | None = None, processed: str | None = None) -> EntityCursor:
    return EntityCursor(
        entity_id=EntityId("entity-1"),
        conversation_id=ConversationId("conv-1"),
        cursor_received=EventId(received) if received else None,
        cursor_processed=EventId(processed) if processed else None,
    )


class TestIsCursorOrderValid:
    def test_both_null_valid(self):
        assert is_cursor_order_valid(_cursor()) is True

    def test_received_set_processed_null_valid(self):
        assert is_cursor_order_valid(_cursor(received=generate_id())) is True

    def test_received_null_processed_set_invalid(self):
        assert is_cursor_order_valid(_cursor(processed=generate_id())) is False

    def test_processed_lte_received_valid(self):
        earlier = generate_id(seed_time=1000000)
        later = generate_id(seed_time=2000000)
        assert is_cursor_order_valid(_cursor(received=later, processed=earlier)) is True

    def test_equal_valid(self):
        id_ = generate_id()
        assert is_cursor_order_valid(_cursor(received=id_, processed=id_)) is True

    def test_processed_gt_received_invalid(self):
        earlier = generate_id(seed_time=1000000)
        later = generate_id(seed_time=2000000)
        assert is_cursor_order_valid(_cursor(received=earlier, processed=later)) is False


class TestHasUnprocessedEvents:
    def test_nothing_received(self):
        assert has_unprocessed_events(_cursor()) is False

    def test_received_no_processed(self):
        assert has_unprocessed_events(_cursor(received=generate_id())) is True

    def test_caught_up(self):
        id_ = generate_id()
        assert has_unprocessed_events(_cursor(received=id_, processed=id_)) is False

    def test_behind(self):
        earlier = generate_id(seed_time=1000000)
        later = generate_id(seed_time=2000000)
        assert has_unprocessed_events(_cursor(received=later, processed=earlier)) is True
