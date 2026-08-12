"""Unit/integration tests for dialectical reasoning engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.analysis.evidence_brief import EvidenceBrief, EvidenceItem
from src.core.dialectics.config import DialecticalMode, DialecticalReasoningConfig, load_dialectical_reasoning_config
from src.core.dialectics.engine import DialecticalEngine
from src.core.dialectics.parse import parse_json_object
from src.core.dialectics.rag_brief import build_principle_cards
from src.core.dialectics.schemas import DialecticalRequest
from src.core.dialectics.shadow import should_sample_shadow, write_shadow_record
from src.core.generation.quality_hooks import apply_quality_post_generate
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig
from tests.helpers.dialectics_mocks import MockBackend

QUOTE = (
    "Монополии срастаются с государственным аппаратом и перекладывают "
    "издержки кризиса на трудящихся через регулирование."
)


def _item(chunk_id: str, stance: str, text: str = QUOTE) -> EvidenceItem:
    return EvidenceItem(
        stance_type=stance,
        source_id=f"src-{chunk_id}",
        source_path=f"path/{chunk_id}",
        chunk_id=chunk_id,
        text=text,
        score=0.9,
        retriever="dense",
        query_used="q",
    )


def _brief(*, with_r3: bool = False) -> EvidenceBrief:
    r3 = [_item("c3", "influence_critical")] if with_r3 else []
    return EvidenceBrief(
        news_title="t",
        news_content="c",
        axes=["монополии"],
        key_concepts=["нефть"],
        r1_core_self=[_item("c1", "core_self")],
        r2_influence_agree=[_item("c2", "influence_agree")],
        r3_influence_critical=r3,
    )


def test_load_mode_default(tmp_path: Path) -> None:
    cfg = load_dialectical_reasoning_config(path=tmp_path / "missing.yaml")
    assert cfg.mode == DialecticalMode.ORCHESTRATION_SINGLE_PASS


def test_parse_fenced_json() -> None:
    raw = '```json\n{"fact": "x", "conclusion": "y"}\n```'
    parsed = parse_json_object(raw)
    assert parsed.status == "parse_ok"
    assert parsed.data is not None
    assert parsed.data["fact"] == "x"


def test_extractive_principle_cards_quote_substring() -> None:
    cfg = DialecticalReasoningConfig()
    cards = build_principle_cards(_brief(), config=cfg)
    assert cards
    assert cards[0].quote in QUOTE or cards[0].quote in cards[0].quote
    assert cards[0].inferred is False


@pytest.mark.asyncio
async def test_engine_valid_publish() -> None:
    cfg = DialecticalReasoningConfig()
    brief = _brief()
    cards = build_principle_cards(brief, config=cfg)
    backend = MockBackend(mode="valid", principle_id=cards[0].principle_id, chunk_id="c1")
    engine = DialecticalEngine(backend=backend, config=cfg)
    result = await engine.analyze(
        request=DialecticalRequest(
            news_title="Правительство регулирует нефтегаз",
            news_content="Введены налоговые меры для отрасли.",
        ),
        brief=brief,
    )
    assert result.outcome in {"publish", "hold_review"}
    assert "Факт:" in result.rendered_text
    assert "анализ опирается" not in result.rendered_text
    assert "r3_absent" in result.reason_codes


@pytest.mark.asyncio
async def test_engine_bad_ids_hold() -> None:
    cfg = DialecticalReasoningConfig()
    backend = MockBackend(mode="bad_ids")
    engine = DialecticalEngine(backend=backend, config=cfg)
    result = await engine.analyze(
        request=DialecticalRequest(news_title="t", news_content="content enough"),
        brief=_brief(),
    )
    assert result.outcome in {"hold_review", "suppress"}
    assert result.quality.passed is False


@pytest.mark.asyncio
async def test_engine_repair_recovers() -> None:
    cfg = DialecticalReasoningConfig(repair_max_attempts=2)
    brief = _brief()
    cards = build_principle_cards(brief, config=cfg)
    backend = MockBackend(
        mode="repair_then_valid",
        principle_id=cards[0].principle_id,
        chunk_id="c1",
    )
    engine = DialecticalEngine(backend=backend, config=cfg)
    result = await engine.analyze(
        request=DialecticalRequest(news_title="t", news_content="новость о регулировании"),
        brief=brief,
        enable_repair=True,
    )
    assert backend.calls >= 2
    assert result.rendered_text


@pytest.mark.asyncio
async def test_engine_missing_brief_suppress() -> None:
    engine = DialecticalEngine(backend=MockBackend(), config=DialecticalReasoningConfig())
    result = await engine.analyze(
        request=DialecticalRequest(news_title="t", news_content="c"),
        brief=None,
    )
    assert result.outcome == "suppress"
    assert "missing_brief" in result.reason_codes


def test_post_qc_skip_structure_keeps_labels() -> None:
    text = "Факт: A\nМеханизм: B потому что C.\nВывод: D"
    out, meta = apply_quality_post_generate(
        text=text,
        chunks=[("c1", 1.0, QUOTE)],
        candidates=[],
        brief=_brief(),
        config=QualityPostcheckConfig(),
        context_has_quotes=True,
        skip_structure_enforce=True,
    )
    assert "Факт:" in out
    assert meta["structure_rebuilt"] is False
    assert meta.get("legacy_stub_rebuild") is False


def test_shadow_sample_and_write(tmp_path: Path) -> None:
    cfg = DialecticalReasoningConfig(shadow_sample_rate=1.0)
    assert should_sample_shadow(cfg)
    from src.core.dialectics.schemas import DialecticalResult

    path = tmp_path / "shadow.jsonl"
    write_shadow_record(
        path=path,
        result=DialecticalResult(outcome="publish", rendered_text="x"),
        news_title="t",
        mode="reasoning_shadow",
        live_text="live",
    )
    assert path.is_file()
    assert "reasoning_shadow" in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_engine_timeout_classified() -> None:
    from src.core.dialectics.pipeline_bridge import run_reasoning_engine

    cfg = DialecticalReasoningConfig(global_timeout_sec=0.05)
    backend = MockBackend(mode="timeout")
    result = await run_reasoning_engine(
        backend=backend,
        config=cfg,
        news_title="t",
        news_content="c",
        brief=_brief(),
        enable_repair=False,
    )
    assert result.outcome == "suppress"
    assert "timeout" in result.reason_codes
