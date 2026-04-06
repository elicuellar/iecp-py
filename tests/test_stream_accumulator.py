"""StreamAccumulator tests -- Phase 8.

Mirrors packages/cli/tests/mcp/stream-accumulator.test.ts exactly.
"""

from __future__ import annotations

import pytest

from iecp_core.cli.mcp.stream_accumulator import StreamAccumulator


class TestStreamAccumulator:
    def setup_method(self) -> None:
        self.acc = StreamAccumulator()

    def test_starts_empty(self) -> None:
        assert self.acc.is_empty() is True
        assert self.acc.get_text() == ""
        assert self.acc.get_chunk_count() == 0

    def test_accumulates_text_from_appends(self) -> None:
        self.acc.append("Hello ")
        self.acc.append("world")
        assert self.acc.get_text() == "Hello world"
        assert self.acc.get_chunk_count() == 2
        assert self.acc.is_empty() is False

    def test_clears_all_state(self) -> None:
        self.acc.append("some text")
        self.acc.clear()
        assert self.acc.is_empty() is True
        assert self.acc.get_text() == ""
        assert self.acc.get_chunk_count() == 0

    def test_tracks_chunk_count_across_multiple_appends(self) -> None:
        self.acc.append("a")
        self.acc.append("b")
        self.acc.append("c")
        assert self.acc.get_chunk_count() == 3

    def test_handles_empty_string_appends(self) -> None:
        self.acc.append("")
        # A chunk was appended, even if empty
        assert self.acc.is_empty() is False
        assert self.acc.get_chunk_count() == 1
        assert self.acc.get_text() == ""
