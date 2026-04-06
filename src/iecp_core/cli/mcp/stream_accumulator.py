"""StreamAccumulator -- tracks stream chunks for a response in progress.

Mirrors packages/cli/src/mcp/StreamAccumulator.ts exactly.
"""

from __future__ import annotations


class StreamAccumulator:
    """Accumulates text chunks for an in-progress streamed response."""

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def append(self, text: str) -> None:
        """Append a text chunk."""
        self._chunks.append(text)

    def get_text(self) -> str:
        """Get the full accumulated text."""
        return "".join(self._chunks)

    def get_chunk_count(self) -> int:
        """Get the number of chunks appended so far."""
        return len(self._chunks)

    def clear(self) -> None:
        """Reset the accumulator."""
        self._chunks = []

    def is_empty(self) -> bool:
        """Check if any chunks have been appended."""
        return len(self._chunks) == 0
