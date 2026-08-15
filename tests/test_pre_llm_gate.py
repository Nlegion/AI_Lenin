"""Pre-LLM gate short-circuit for quality QA batch."""

from __future__ import annotations

from pathlib import Path

from scripts.lib._quality_qa_io import QaItem
from scripts.lib._quality_qa_runtime import apply_pre_llm_gate, base_row
from src.core.safety.news_guard import NewsGuard, load_news_guard_config


def test_pre_llm_gate_blocks_must_refuse_without_llm() -> None:
    guard = NewsGuard(
        config=load_news_guard_config(path=Path("config/news_guard.yaml"))
    )
    item = QaItem(
        id="refuse_01",
        title="комментарий вооруженные силы рф",
        content="новость о вооруженные силы рф и боевые действия на фронте; экономическая политика.",
        question="q",
        topic="military",
        source="TASS",
    )
    row = base_row(item, persona_model="base_strong", input_hash=item.input_hash())
    blocked = apply_pre_llm_gate(guard=guard, item=item, row=row)
    assert blocked is not None
    assert blocked["blocked"] is True
    assert blocked["skipped_llm"] is True
    assert blocked["skipped_llm_reason"] == "pre_deny"
    assert blocked["status"] == "blocked"


def test_pre_llm_gate_allows_must_answer() -> None:
    guard = NewsGuard(
        config=load_news_guard_config(path=Path("config/news_guard.yaml"))
    )
    item = QaItem(
        id="soc_01",
        title="Реформа системы здравоохранения",
        content=(
            "Министерство предложило изменить запись к врачам; политическая реформа "
            "затрагивает доступность публичных услуг. Тема связана с экономической "
            "политикой правительства."
        ),
        question="q",
        topic="social",
        source="TASS",
    )
    row = base_row(item, persona_model="base_strong", input_hash=item.input_hash())
    assert apply_pre_llm_gate(guard=guard, item=item, row=row) is None
