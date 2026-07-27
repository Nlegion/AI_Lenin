"""Anti-cliché gate unit tests."""

from __future__ import annotations

from src.core.safety.cliche_gate import cliche_gate


def test_cliche_gate_flags_low_r1_overlap():
    result = cliche_gate(
        analysis="революция эксплуатация пролетариат буржуазия классовая борьба",
        r1_text="совершенно другой текст без пересечения",
        r1_count=0,
        warn_only=True,
    )
    assert "r1_count_zero" in result.reason_codes
    assert result.r1_jaccard < 0.05
