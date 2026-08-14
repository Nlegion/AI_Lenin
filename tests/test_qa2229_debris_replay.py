"""Replay scrub on live QA 20260813-2229 answers; assert listed debris patterns are gone."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.core.generation.answer_body_cleanup import cleanup_answer_body
from src.core.generation.output_artifacts import final_public_scrub
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

ROOT = Path(__file__).resolve().parents[1]
JSONL = (
    ROOT
    / ".cursor"
    / "artifacts"
    / "quality"
    / "live_news_qa_50_20260813-2229_20260813-2229.jsonl"
)

_BROKEN_STANCE = re.compile(
    r"(?i)(?:ленин|lenin)\s*\([^)]*core_[^)]*\)",
)
_MD_DEBRIS = re.compile(r"(?:---\s*){2,}|(?:##\s*){2,}|---\s*##|##\s*---")
_PROMPT_TASK = re.compile(r"(?i)задача\s*:\s*краткий\s+анализ")
_MESTO = re.compile(r"«?\s*\[(?:место|обезличено)\]\s*»?", re.IGNORECASE)
_BOLD_SECTION = re.compile(
    r"(?i)\*{1,2}(?:факт|механизм|вывод)\*{0,2}\s*:",
)


def _successful_answers(path: Path) -> list[str]:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("status") or "").casefold() != "done":
            continue
        if row.get("blocked"):
            continue
        answer = str(row.get("answer") or "").strip()
        if not answer:
            continue
        rows.append(answer)
    return rows


@pytest.mark.skipif(not JSONL.is_file(), reason="QA-2229 jsonl artifact missing")
def test_replay_scrub_clears_qa2229_debris_patterns() -> None:
    cfg = QualityPostcheckConfig()
    answers = _successful_answers(JSONL)
    assert answers, "expected successful answers in QA-2229 jsonl"

    hits = {
        "broken_stance": 0,
        "md_debris": 0,
        "prompt_task": 0,
        "mesto": 0,
        "bold_section": 0,
    }
    for raw in answers:
        body = cleanup_answer_body(text=raw, config=cfg)
        cleaned, _codes = final_public_scrub(body.text)
        if _BROKEN_STANCE.search(cleaned):
            hits["broken_stance"] += 1
        if _MD_DEBRIS.search(cleaned) or re.search(
            r"(?:\s*(?:---|##))+\s*\.?\s*$", cleaned
        ):
            hits["md_debris"] += 1
        if _PROMPT_TASK.search(cleaned):
            hits["prompt_task"] += 1
        if _MESTO.search(cleaned):
            hits["mesto"] += 1
        if _BOLD_SECTION.search(cleaned):
            hits["bold_section"] += 1

    assert hits == {
        "broken_stance": 0,
        "md_debris": 0,
        "prompt_task": 0,
        "mesto": 0,
        "bold_section": 0,
    }, hits
