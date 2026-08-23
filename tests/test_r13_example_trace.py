"""Tests for R1/R2/R3 example trace report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.lib._r13_example_report import (
    format_report,
    format_slot_markdown,
    load_fixture_qa_items,
    load_jsonl_qa_items,
    write_report_files,
)
from src.core.analysis.evidence_brief import (
    EvidenceBrief,
    EvidenceItem,
    items_trace_payload,
)
from src.core.generation.pipeline import _rag_stats_from_brief


def _item(
    *, stance: str, chunk_id: str, text: str, path: str = "pss/v27.txt"
) -> EvidenceItem:
    return EvidenceItem(
        stance_type=stance,
        source_id="src",
        source_path=path,
        chunk_id=chunk_id,
        text=text,
        score=0.42,
        retriever="dense",
        query_used="q",
    )


def test_items_trace_payload_clips_and_flags_truncation():
    payload = items_trace_payload(
        [_item(stance="core_self", chunk_id="c1", text="x" * 50)],
        text_cap=10,
    )
    assert len(payload) == 1
    assert payload[0]["truncated"] is True
    assert str(payload[0]["text"]).endswith("…")
    assert payload[0]["chunk_id"] == "c1"
    assert payload[0]["stance_type"] == "core_self"


def test_rag_stats_split_slots():
    brief = EvidenceBrief(
        news_title="t",
        news_content="c",
        axes=[],
        key_concepts=[],
        r1_core_self=[_item(stance="core_self", chunk_id="r1", text="lenin quote")],
        r2_influence_agree=[
            _item(stance="influence_agree", chunk_id="r2", text="marx")
        ],
        r3_influence_critical=[],
    )
    stats = _rag_stats_from_brief(brief)
    assert stats["r1_count"] == 1
    assert stats["r2_count"] == 1
    assert stats["r3_count"] == 0
    assert stats["r1_items"][0]["text"] == "lenin quote"
    assert stats["r2_items"][0]["text"] == "marx"
    assert stats["r3_items"] == []


def test_format_slot_empty_is_explicit():
    text = format_slot_markdown(title="R3 — Критика", items=[])
    assert "(пусто)" in text
    assert "— 0" in text


def test_format_report_includes_news_slots_and_answer():
    report = format_report(
        [
            {
                "id": "n1",
                "title": "Рост инфляции",
                "content": "Правительство обсуждает цены.",
                "source": "TASS",
                "status": "done",
                "orchestration_mode": "dialectical_v1",
                "r1_items": [
                    {"source_path": "pss/v27", "score": 0.5, "text": "кризис"}
                ],
                "r2_items": [],
                "r3_items": [],
                "answer": "Факт: инфляция. Механизм: капитал. Вывод: класс.",
            }
        ]
    )
    assert "Рост инфляции" in report
    assert "Правительство обсуждает цены." in report
    assert "R1 — Ленин" in report
    assert "кризис" in report
    assert "(пусто)" in report
    assert "Факт: инфляция" in report


def test_load_fixtures_and_write_report(tmp_path: Path):
    items = load_fixture_qa_items(path=Path("config/dryrun_fixtures.yaml"), limit=2)
    assert len(items) == 2
    assert items[0].title
    jsonl = tmp_path / "in.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "id": "a1",
                "title": "t",
                "content": "c",
                "question": "q",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_jsonl_qa_items(path=jsonl, limit=1)
    assert loaded[0].id == "a1"
    md_path, jsonl_path = write_report_files(
        output_dir=tmp_path,
        stem="r13_example_trace",
        rows=[{"id": "a1", "title": "t", "content": "c", "answer": "ok"}],
    )
    assert md_path.exists()
    assert jsonl_path.exists()
    assert "### Новость" in md_path.read_text(encoding="utf-8")
    assert "ok" in jsonl_path.read_text(encoding="utf-8")
