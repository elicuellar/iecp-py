"""Output Filter tests -- Phase 6: Artificer Runtime (§11 of the IECP spec)."""

from __future__ import annotations

import pytest

from iecp_core.artificer import ArtificerPersona, OutputFilter, OutputFilterConfig


PERSONA = ArtificerPersona(
    name="Meina",
    role="analyst",
    phase="Discovery",
    system_prompt="You are Meina, an analyst.",
)


class TestOutputFilter:
    def setup_method(self) -> None:
        self.filter = OutputFilter()

    def test_passes_valid_output(self) -> None:
        result = self.filter.check("This is a valid response.", PERSONA)
        assert result is None

    def test_rejects_empty_output(self) -> None:
        result = self.filter.check("", PERSONA)
        assert result == "Output is empty"

    def test_rejects_whitespace_only_output(self) -> None:
        result = self.filter.check("   \n\t  ", PERSONA)
        assert result == "Output is empty"

    def test_rejects_over_length_output(self) -> None:
        long_filter = OutputFilter(OutputFilterConfig(max_length=50))
        result = long_filter.check("a" * 51, PERSONA)
        assert result is not None
        assert "exceeds maximum length" in result

    def test_rejects_impersonation_attempt(self) -> None:
        result = self.filter.check("[Alice]: I think we should proceed.", PERSONA)
        assert result is not None
        assert "impersonate" in result
        assert "Alice" in result

    def test_passes_mention_mid_text(self) -> None:
        result = self.filter.check(
            "I agree with [Alice]: she makes a good point.", PERSONA
        )
        assert result is None

    def test_passes_own_name_prefix(self) -> None:
        result = self.filter.check("[Meina]: Here is my analysis.", PERSONA)
        assert result is None

    def test_respects_custom_max_length(self) -> None:
        short_filter = OutputFilter(OutputFilterConfig(max_length=10))
        assert short_filter.check("short", PERSONA) is None
        assert short_filter.check("this is too long", PERSONA) is not None
        assert "exceeds maximum length" in short_filter.check("this is too long", PERSONA)  # type: ignore[arg-type]
