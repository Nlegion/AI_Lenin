"""Tests for live PreRagCensor gate parity."""

from __future__ import annotations

import asyncio
from pathlib import Path

from scripts._live_news_qa_censor import apply_live_pre_rag_gate, build_live_pre_rag_censor
from scripts._live_news_qa_fetch import news_row_to_qa_item
from scripts._quality_qa_runtime import base_row


def _run(coro):
    return asyncio.run(coro)


def test_live_pre_rag_blocks_sport() -> None:
    censor = build_live_pre_rag_censor(base_dir=Path("."), enable_memory_cache=True)
    item = news_row_to_qa_item(
        {
            "id": "sport1",
            "title": "Пловец Кожакин завоевал серебро ЧЕ",
            "content": "Россиянин стал вторым в финальном заплыве чемпионата Европы.",
            "source": "TASS",
        }
    )
    row = base_row(item, persona_model="base_strong", input_hash=item.input_hash())
    outcome = _run(apply_live_pre_rag_gate(censor=censor, item=item, row=row))
    assert outcome.blocked_row is not None
    assert outcome.blocked_row["censor_decision"] in {"hard_block", "skip"}
    assert outcome.blocked_row["skipped_llm_reason"] in {"pre_deny", "out_of_scope_skip"}


def test_live_pre_rag_blocks_fire() -> None:
    censor = build_live_pre_rag_censor(base_dir=Path("."), enable_memory_cache=True)
    item = news_row_to_qa_item(
        {
            "id": "fire1",
            "title": "В Петербурге при пожаре в ДК повреждены помещения",
            "content": "Никто не пострадал при пожаре в доме культуры.",
            "source": "TASS",
        }
    )
    row = base_row(item, persona_model="base_strong", input_hash=item.input_hash())
    outcome = _run(apply_live_pre_rag_gate(censor=censor, item=item, row=row))
    assert outcome.blocked_row is not None
    assert outcome.blocked_row["censor_decision"] == "hard_block"


def test_live_pre_rag_blocks_war_markers() -> None:
    censor = build_live_pre_rag_censor(base_dir=Path("."), enable_memory_cache=True)
    item = news_row_to_qa_item(
        {
            "id": "war1",
            "title": "В Белгородской области при атаках ВСУ погиб человек",
            "content": "Еще двое пострадали после удара БПЛА.",
            "source": "TASS",
        }
    )
    row = base_row(item, persona_model="base_strong", input_hash=item.input_hash())
    outcome = _run(apply_live_pre_rag_gate(censor=censor, item=item, row=row))
    assert outcome.blocked_row is not None
    assert outcome.blocked_row["censor_decision"] == "hard_block"


def test_live_pre_rag_review_passes_yellow_context() -> None:
    censor = build_live_pre_rag_censor(base_dir=Path("."), enable_memory_cache=True)
    # Force a low-signal political item that is not sport/fire/war hard-block.
    item = news_row_to_qa_item(
        {
            "id": "rev1",
            "title": "Эксперт прокомментировал рост тарифов на электроэнергию",
            "content": "Аналитик отметил влияние регулирования рынка на потребителей.",
            "source": "TASS",
        }
    )
    row = base_row(item, persona_model="base_strong", input_hash=item.input_hash())
    outcome = _run(apply_live_pre_rag_gate(censor=censor, item=item, row=row, strict_review=False))
    if outcome.generation is not None:
        assert outcome.generation.censor_decision in {"allow", "review"}
        assert outcome.generation.risk_tier in {"green", "yellow"}
    else:
        assert outcome.blocked_row is not None
