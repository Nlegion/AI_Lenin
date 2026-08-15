"""Unit tests for quote allowlist, grounding, loops, fact/opinion (Quality QA plan)."""

from __future__ import annotations

from pathlib import Path

from src.core.generation.loop_detect import detect_and_fix_loops
from src.core.generation.quote_allowlist import (
    extract_quote_candidates,
    quote_allowlist_present,
)
from src.core.generation.quote_postcheck import (
    apply_quote_postcheck,
    check_critical_attribution,
)
from src.core.generation.text_normalize import normalize_for_grounding
from src.core.safety.fact_opinion import needs_fact_opinion_extra
from src.core.settings.quality_postcheck_config import load_quality_postcheck_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_quality_postcheck_config(path=ROOT / "config" / "quality_postcheck.yaml")


def test_normalize_unifies_quotes_and_yo() -> None:
    a = normalize_for_grounding("«Ёжик» — тест")
    b = normalize_for_grounding('"ежик" - тест')
    assert a == b


def test_extract_candidates_from_chunk_quotes() -> None:
    chunks = [
        (
            "c1",
            1.0,
            "Ленин писал: «Империализм есть канун социальной революции пролетариата». Далее текст.",
        ),
    ]
    cands = extract_quote_candidates(chunks=chunks, config=CFG)
    assert quote_allowlist_present(cands)
    assert any("империализм" in c.text.lower() for c in cands)


def test_trivial_stoplist_rejects_lead_ins() -> None:
    chunks = [("c1", 1.0, "«Как сообщается»")]
    cands = extract_quote_candidates(chunks=chunks, config=CFG)
    assert cands == []


def test_ungrounded_quote_stripped() -> None:
    chunks = [("c1", 1.0, "«Реальный фрагмент из корпуса о монополиях и банках».")]
    cands = extract_quote_candidates(chunks=chunks, config=CFG)
    answer = (
        "Событие показывает монополии банковского капитала в экономике. "
        "«вся жизнь есть борьба» — Ленин, *О спорте*, том 42, стр. 93."
    )
    result = apply_quote_postcheck(text=answer, candidates=cands, config=CFG)
    assert "вся жизнь есть борьба" not in result.text
    assert "монополии" in result.text
    assert result.quote_removed or result.critical_attribution_hallucination
    assert result.used_static_template is False


def test_critical_attribution_invented_volume() -> None:
    codes = check_critical_attribution(
        "Как писал Ленин, том 99, стр. 1.", candidates=[]
    )
    assert any("volume" in c or "page" in c for c in codes)


def test_path_leak_scrubbed() -> None:
    text = "См. [source: data/pss/том 19.txt] в анализе."
    result = apply_quote_postcheck(text=text, candidates=[], config=CFG)
    assert "/pss/" not in result.text
    assert "[source:" not in result.text.lower()


def test_loop_drop_duplicate_paragraph() -> None:
    para = "Военная диктатура усиливает репрессивные меры против рабочего класса в этом районе."
    text = f"{para}\n\n{para}\n\nЕщё один тезис о монополиях."
    enabled = CFG.model_copy(update={"loop_fix_enabled": True})
    result = detect_and_fix_loops(text, config=enabled, rag_empty=False)
    assert result.loop_detected
    assert result.loop_action == "drop_duplicate_paragraph"
    assert result.text.count(para) == 1


def test_empty_rag_loop_dedupes_not_template() -> None:
    para = "Общий принцип без опоры на факты новости повторяется снова и снова здесь."
    text = f"{para}\n\n{para}"
    enabled = CFG.model_copy(update={"loop_fix_enabled": True})
    result = detect_and_fix_loops(text, config=enabled, rag_empty=True)
    assert result.loop_action == "drop_duplicate_paragraph"
    assert "Недостаточно данных" not in result.text
    assert result.text.count(para) == 1


def test_fact_opinion_expert_subject() -> None:
    assert needs_fact_opinion_extra(
        title="Эксперт Байдильдинов",
        content="Эксперт Байдильдинов заявил, что рынок нефти обвалится.",
    )
    assert not needs_fact_opinion_extra(
        title="Минфин",
        content="Минфин сообщил о исполнении бюджета за квартал.",
    )
