"""Output Filter -- §11 of the IECP specification.

Validates artificer output before it's committed to the event log.
Returns None if valid, or a rejection reason string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import ArtificerPersona


# -- Configuration ------------------------------------------------------------


@dataclass
class OutputFilterConfig:
    """Configuration for the OutputFilter."""

    max_length: int = 10_000
    """Maximum response length in characters."""


# -- Impersonation Pattern ----------------------------------------------------

# Matches `[Name]:` at the start of the output.
# Used to detect when an artificer tries to speak as another entity.
_IMPERSONATION_PATTERN = re.compile(r"^\[([^\]]+)\]:")


# -- OutputFilter -------------------------------------------------------------


class OutputFilter:
    """Validates artificer output before commit."""

    def __init__(self, config: OutputFilterConfig | None = None) -> None:
        self._config = config or OutputFilterConfig()

    def check(self, text: str, persona: ArtificerPersona) -> str | None:
        """Check artificer output before commit.

        Returns None if the output passes all checks, or a rejection reason
        string.
        """
        # 1. Not empty
        if not text or not text.strip():
            return "Output is empty"

        # 2. Max length
        if len(text) > self._config.max_length:
            return f"Output exceeds maximum length of {self._config.max_length} characters"

        # 3. No impersonation -- response must not start with another entity's
        #    name prefix
        match = _IMPERSONATION_PATTERN.match(text)
        if match:
            detected_name = match.group(1)
            if detected_name != persona.name:
                return f'Output appears to impersonate "{detected_name}"'

        return None
