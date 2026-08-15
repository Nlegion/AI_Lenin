"""Stage 2 soft-repair: keep analysis body; hard template only when configured."""

from __future__ import annotations

from pathlib import Path

from src.core.generation.loop_detect import detect_and_fix_loops
from src.core.generation.output_artifacts import (
    apply_artifact_pass,
    detect_encoding_artifacts,
)
from src.core.generation.quote_allowlist import extract_quote_candidates
from src.core.generation.quote_postcheck import apply_quote_postcheck
from src.core.settings.quality_postcheck_config import (
    QualityPostcheckConfig,
    load_quality_postcheck_config,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = load_quality_postcheck_config(path=ROOT / "config" / "quality_postcheck.yaml")


def test_stage0_flags_disable_template_escalation() -> None:
    # Stage config is read from repo YAML; keep assertion aligned with active policy.
    assert CFG.loop_fix_enabled is True
    assert CFG.yellow_output_filter_enabled is False
    assert CFG.quote_postcheck_enforce_mode == "soft"
    assert CFG.artifact_enforce_mode == "soft"
    assert CFG.postprocess_clean_mode == "live"
    assert CFG.trial50_hotfixes.get("generation_hotfixes_enabled") is True
    assert CFG.trial50_hotfixes.get("loop_strip_enabled") is True


def test_ungrounded_quote_keeps_analysis_body() -> None:
    chunks = [("c1", 1.0, "«Реальный фрагмент из корпуса о монополиях и банках».")]
    cands = extract_quote_candidates(chunks=chunks, config=CFG)
    answer = (
        "Событие показывает концентрацию капитала в банковском секторе. "
        "«вся жизнь есть борьба» — Ленин, том 42, стр. 93."
    )
    result = apply_quote_postcheck(text=answer, candidates=cands, config=CFG)
    assert "вся жизнь есть борьба" not in result.text
    assert "концентрацию капитала" in result.text
    assert result.used_static_template is False


def test_loop_dedupe_without_static_insufficient() -> None:
    para = "Общий принцип без опоры на факты новости повторяется снова и снова здесь."
    text = f"{para}\n\n{para}"
    result = detect_and_fix_loops(text, config=CFG, rag_empty=True)
    assert result.loop_detected is True
    assert result.loop_action == "drop_duplicate_paragraph"
    assert "Недостаточно данных" not in result.text


def test_encoding_detect_without_fallback_in_soft_mode() -> None:
    text = "Швейцарские СЃ сообщили о переговорах по экономике и тарифам."
    assert "artifact:mojibake_sg" in detect_encoding_artifacts(text)
    res = apply_artifact_pass(text=text, config=CFG, item_id="enc1")
    assert res.used_fallback is False
    assert "переговорах" in res.text


def test_hard_mode_still_allows_fallback() -> None:
    hard = QualityPostcheckConfig(
        artifact_enforce_mode="strict",
        hard_fallback_on_broken_output=True,
        static_safe_template="Не удалось сформировать корректный анализ по данной новости.",
        fallback_templates=[
            "Не удалось сформировать корректный анализ по данной новости."
        ],
    )
    text = "Швейцарские СЃ сообщили."
    # Force encoding scrubber path via hotfix flags (defaults true).
    res = apply_artifact_pass(text=text, config=hard, item_id="enc2")
    assert res.used_fallback is True
