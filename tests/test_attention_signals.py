"""Tests for AttentionSignalManager -- SS18 of the specification."""

from unittest.mock import patch
import time

import pytest

from iecp_core.signals import AttentionSignalManager, AttentionSignalConfig
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId, BatchId

ENTITY_A = EntityId("entity-a")
ENTITY_B = EntityId("entity-b")
ENTITY_C = EntityId("entity-c")
CONV_1 = ConversationId("conv-1")
CONV_2 = ConversationId("conv-2")
BATCH_1 = BatchId("batch-1")
BATCH_2 = BatchId("batch-2")


class TestAttentionSignalManager:
    def test_register_signal_successfully(self) -> None:
        manager = AttentionSignalManager()
        try:
            result = manager.signal(
                entity_id=ENTITY_A,
                conversation_id=CONV_1,
                signal_type="thinking",
                batch_id=BATCH_1,
                note="processing the request",
            )
            assert result is True

            signal = manager.get_entity_signal(CONV_1, ENTITY_A)
            assert signal is not None
            assert signal.entity_id == ENTITY_A
            assert signal.signal_type == "thinking"
            assert signal.note == "processing the request"
            assert signal.batch_id == BATCH_1
        finally:
            manager.destroy()

    def test_rate_limit_1_signal_per_batch_per_entity(self) -> None:
        manager = AttentionSignalManager()
        try:
            first = manager.signal(
                entity_id=ENTITY_A,
                conversation_id=CONV_1,
                signal_type="thinking",
                batch_id=BATCH_1,
            )
            assert first is True

            second = manager.signal(
                entity_id=ENTITY_A,
                conversation_id=CONV_1,
                signal_type="acknowledged",
                batch_id=BATCH_1,
            )
            assert second is False

            # Same entity, different batch should work
            third = manager.signal(
                entity_id=ENTITY_A,
                conversation_id=CONV_1,
                signal_type="acknowledged",
                batch_id=BATCH_2,
            )
            assert third is True
        finally:
            manager.destroy()

    def test_replace_older_signal_same_entity_conversation(self) -> None:
        manager = AttentionSignalManager()
        try:
            # No batch_id -> no rate limiting
            manager.signal(
                entity_id=ENTITY_A,
                conversation_id=CONV_1,
                signal_type="listening",
            )
            signal = manager.get_entity_signal(CONV_1, ENTITY_A)
            assert signal is not None
            assert signal.signal_type == "listening"

            manager.signal(
                entity_id=ENTITY_A,
                conversation_id=CONV_1,
                signal_type="thinking",
            )
            signal = manager.get_entity_signal(CONV_1, ENTITY_A)
            assert signal is not None
            assert signal.signal_type == "thinking"
        finally:
            manager.destroy()

    def test_expire_signals_after_ttl_ms(self) -> None:
        manager = AttentionSignalManager(AttentionSignalConfig(ttl_ms=1000))
        try:
            fake_now = 1000.0  # seconds
            with patch("iecp_core.signals.attention_signal_manager.time") as mock_time:
                mock_time.time.return_value = fake_now
                manager.signal(
                    entity_id=ENTITY_A,
                    conversation_id=CONV_1,
                    signal_type="thinking",
                )

                assert manager.get_entity_signal(CONV_1, ENTITY_A) is not None

                # Advance past TTL (1000ms = 1s)
                mock_time.time.return_value = fake_now + 1.001
                assert manager.get_entity_signal(CONV_1, ENTITY_A) is None
        finally:
            manager.destroy()

    def test_clear_expired_removes_only_expired(self) -> None:
        manager = AttentionSignalManager(AttentionSignalConfig(ttl_ms=1000))
        try:
            fake_now = 1000.0
            with patch("iecp_core.signals.attention_signal_manager.time") as mock_time:
                mock_time.time.return_value = fake_now
                manager.signal(
                    entity_id=ENTITY_A,
                    conversation_id=CONV_1,
                    signal_type="thinking",
                )

                mock_time.time.return_value = fake_now + 0.5
                manager.signal(
                    entity_id=ENTITY_B,
                    conversation_id=CONV_1,
                    signal_type="listening",
                )

                # A expired (1100ms), B still alive (600ms)
                mock_time.time.return_value = fake_now + 1.1
                manager.clear_expired()

                signals = manager.get_signals(CONV_1)
                assert len(signals) == 1
                assert signals[0].entity_id == ENTITY_B
        finally:
            manager.destroy()

    def test_multiple_entities_signal_same_conversation(self) -> None:
        manager = AttentionSignalManager()
        try:
            manager.signal(
                entity_id=ENTITY_A,
                conversation_id=CONV_1,
                signal_type="thinking",
            )
            manager.signal(
                entity_id=ENTITY_B,
                conversation_id=CONV_1,
                signal_type="acknowledged",
            )
            manager.signal(
                entity_id=ENTITY_C,
                conversation_id=CONV_1,
                signal_type="deferred",
                note="will follow up",
            )

            signals = manager.get_signals(CONV_1)
            assert len(signals) == 3

            types = sorted(s.signal_type for s in signals)
            assert types == ["acknowledged", "deferred", "thinking"]
        finally:
            manager.destroy()

    def test_get_signals_only_for_specified_conversation(self) -> None:
        manager = AttentionSignalManager()
        try:
            manager.signal(
                entity_id=ENTITY_A,
                conversation_id=CONV_1,
                signal_type="thinking",
            )
            manager.signal(
                entity_id=ENTITY_A,
                conversation_id=CONV_2,
                signal_type="listening",
            )

            conv1_signals = manager.get_signals(CONV_1)
            assert len(conv1_signals) == 1
            assert conv1_signals[0].conversation_id == CONV_1

            conv2_signals = manager.get_signals(CONV_2)
            assert len(conv2_signals) == 1
            assert conv2_signals[0].conversation_id == CONV_2
        finally:
            manager.destroy()

    def test_get_entity_signal_for_specific_entity(self) -> None:
        manager = AttentionSignalManager()
        try:
            manager.signal(
                entity_id=ENTITY_A,
                conversation_id=CONV_1,
                signal_type="thinking",
            )
            manager.signal(
                entity_id=ENTITY_B,
                conversation_id=CONV_1,
                signal_type="acknowledged",
            )

            signal_a = manager.get_entity_signal(CONV_1, ENTITY_A)
            assert signal_a is not None
            assert signal_a.signal_type == "thinking"

            signal_b = manager.get_entity_signal(CONV_1, ENTITY_B)
            assert signal_b is not None
            assert signal_b.signal_type == "acknowledged"

            # Non-existent entity
            signal_c = manager.get_entity_signal(CONV_1, ENTITY_C)
            assert signal_c is None
        finally:
            manager.destroy()

    def test_clear_signal_for_specific_entity(self) -> None:
        manager = AttentionSignalManager()
        try:
            manager.signal(
                entity_id=ENTITY_A,
                conversation_id=CONV_1,
                signal_type="thinking",
            )
            manager.signal(
                entity_id=ENTITY_B,
                conversation_id=CONV_1,
                signal_type="acknowledged",
            )

            manager.clear_signal(CONV_1, ENTITY_A)

            assert manager.get_entity_signal(CONV_1, ENTITY_A) is None
            assert manager.get_entity_signal(CONV_1, ENTITY_B) is not None
        finally:
            manager.destroy()

    def test_clean_up_on_destroy(self) -> None:
        manager = AttentionSignalManager()
        manager.signal(
            entity_id=ENTITY_A,
            conversation_id=CONV_1,
            signal_type="thinking",
        )

        manager.destroy()

        result = manager.signal(
            entity_id=ENTITY_B,
            conversation_id=CONV_1,
            signal_type="listening",
        )
        assert result is False
        assert manager.get_signals(CONV_1) == []
