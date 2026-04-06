"""SimpleTokenEstimator -- Character-based token estimation.

Uses the rough heuristic of 4 characters ~ 1 token.
Designed to be swapped out for a real tokenizer (tiktoken, etc.) later.
"""

from __future__ import annotations

import json
import math

from ..types.event import Event

CHARS_PER_TOKEN = 4


class SimpleTokenEstimator:
    """Simple token estimator based on character count.

    4 characters ~ 1 token. Good enough for budget calculations
    where precision isn't critical.
    """

    def estimate(self, text: str) -> int:
        if len(text) == 0:
            return 0
        return math.ceil(len(text) / CHARS_PER_TOKEN)

    def estimate_event(self, event: Event) -> int:
        serialized = json.dumps(event.model_dump(), default=str)
        return self.estimate(serialized)
