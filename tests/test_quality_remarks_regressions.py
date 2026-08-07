"""Gate expectations from quality remarks table IDs."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.safety.fact_opinion import needs_fact_opinion_extra
from src.core.safety.news_guard import NewsGuard

ROOT = Path(__file__).resolve().parents[1]
GUARD = NewsGuard.from_file(ROOT / "config" / "news_guard.yaml")


def test_remarks_gate_expectations() -> None:
    path = ROOT / "data" / "eval" / "quality_remarks_regressions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if "expect_decision" in row:
            result = GUARD.evaluate_input(title=row["title"], content=row["content"], source="TASS")
            assert result.decision == row["expect_decision"], (
                f"{row['id']}: got {result.decision} ({result.reason})"
            )
        if row.get("expect_fact_opinion"):
            assert needs_fact_opinion_extra(title=row["title"], content=row["content"])
