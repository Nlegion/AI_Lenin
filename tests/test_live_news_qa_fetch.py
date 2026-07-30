"""Tests for live-news QA fetch adapter."""

from __future__ import annotations

from scripts._live_news_qa_fetch import news_row_to_qa_item


def test_news_row_to_qa_item() -> None:
    item = news_row_to_qa_item(
        {
            "id": "abc123",
            "title": "Рост инфляции",
            "content": "Краткое описание новости.",
            "source": "TASS",
        }
    )
    assert item.id == "abc123"
    assert item.source == "TASS"
    assert item.topic == "live"
    assert "Рост инфляции" in item.question
    assert item.content.startswith("Краткое")
