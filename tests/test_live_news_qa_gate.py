"""Tests for live-news soft-pass gate and PII false-positive fix."""

from __future__ import annotations

from pathlib import Path

from scripts._live_news_qa_fetch import news_row_to_qa_item
from scripts._live_news_qa_gate import apply_live_pre_llm_gate
from scripts._quality_qa_runtime import base_row
from src.core.safety.news_guard import NewsGuard, load_news_guard_config


def test_live_soft_pass_unknown_quarantine() -> None:
    guard = NewsGuard(config=load_news_guard_config(path=Path("config/news_guard.yaml")))
    item = news_row_to_qa_item(
        {
            "id": "x1",
            "title": "Запуск нового маршрута метро",
            "content": "Город открыл станцию на окраине.",
            "source": "TASS",
        }
    )
    row = base_row(item, persona_model="base_strong", input_hash=item.input_hash())
    blocked = apply_live_pre_llm_gate(guard=guard, item=item, row=row, unknown_as_allow=True)
    assert blocked is None
    assert row.get("gate_soft_pass") == "unknown_no_allow_topic"


def test_live_still_blocks_military() -> None:
    guard = NewsGuard(config=load_news_guard_config(path=Path("config/news_guard.yaml")))
    item = news_row_to_qa_item(
        {
            "id": "x2",
            "title": "комментарий вооруженные силы рф",
            "content": "новость о вооруженные силы рф и боевые действия; экономическая политика.",
            "source": "TASS",
        }
    )
    row = base_row(item, persona_model="base_strong", input_hash=item.input_hash())
    blocked = apply_live_pre_llm_gate(guard=guard, item=item, row=row, unknown_as_allow=True)
    assert blocked is not None
    assert blocked["skipped_llm"] is True


def test_news_gate_pii_does_not_match_lowercase_three_words() -> None:
    guard = NewsGuard(config=load_news_guard_config(path=Path("config/news_guard.yaml")))
    result = guard.evaluate_input(
        title="запуск нового маршрута",
        content="город открыл станцию на окраине без фамилий.",
        source="TASS",
    )
    assert result.reason != "private pii detected without public-interest context"
