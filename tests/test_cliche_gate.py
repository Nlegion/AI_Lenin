"""Anti-cliché gate unit tests (decision table)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.safety.anti_cliche_config import AntiClicheConfig
from src.core.safety.cliche_gate import cliche_gate
from src.core.settings.gate_constants import (
    CLICHE_CODE_LEXICON_DENSE,
    CLICHE_CODE_LOW_R1_OVERLAP,
    CLICHE_CODE_NO_R1,
    CLICHE_CODE_SKIPPED_NO_BRIEF,
)

REPO = Path(__file__).resolve().parents[1]
CASES_PATH = REPO / "data" / "eval" / "anti_cliche_cases.jsonl"


def _load_cases() -> list[dict]:
    rows: list[dict] = []
    with CASES_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["id"])
def test_cliche_gate_jsonl_cases(case: dict) -> None:
    result = cliche_gate(
        analysis=case["analysis"],
        brief_present=bool(case["brief_present"]),
        r1_text=case.get("r1_text", ""),
        r1_count=int(case.get("r1_count", 0)),
    )
    assert result.reason_codes == case["expect_codes"]
    if case.get("expect_skipped"):
        assert result.skipped is True


def test_both_codes_when_both_signals() -> None:
    result = cliche_gate(
        analysis="революция эксплуатация пролетариат буржуазия классовая диктатура",
        brief_present=True,
        r1_text="тарифы логистика зерно поставки",
        r1_count=2,
    )
    assert CLICHE_CODE_LOW_R1_OVERLAP in result.reason_codes
    assert CLICHE_CODE_LEXICON_DENSE in result.reason_codes


def test_quote_anchor_does_not_clear_no_r1() -> None:
    result = cliche_gate(
        analysis="Согласно данным, революция эксплуатация пролетариат буржуазия классовая",
        brief_present=True,
        r1_text="",
        r1_count=0,
    )
    assert CLICHE_CODE_NO_R1 in result.reason_codes


def test_brief_none_skips() -> None:
    result = cliche_gate(
        analysis="революция эксплуатация пролетариат",
        brief_present=False,
        r1_count=0,
    )
    assert result.skipped is True
    assert CLICHE_CODE_SKIPPED_NO_BRIEF in result.reason_codes
    assert result.blocked is False


def test_fail_open_on_bad_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr(
        "src.core.safety.cliche_gate._evaluate_cliche_gate",
        _boom,
    )
    result = cliche_gate(
        analysis="x",
        brief_present=True,
        r1_count=1,
        r1_text="y",
        config=AntiClicheConfig(),
    )
    assert result.skipped is True
    assert result.skip_reason and result.skip_reason.startswith("error:")
