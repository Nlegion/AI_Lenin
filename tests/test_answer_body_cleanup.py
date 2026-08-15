from __future__ import annotations

from pathlib import Path

from src.core.generation.answer_body_cleanup import cleanup_answer_body
from src.core.generation.output_artifacts import apply_artifact_pass
from src.core.generation.publishability import is_publishable_analysis
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "answer_postprocess"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_scrub_stance_and_trailing_exact_fact() -> None:
    body = _load("stance_trailing_fact_bsw.in.txt")
    result = cleanup_answer_body(text=body)
    assert "Ленин (agreement)" not in result.text
    assert "Ленин (disagreement)" not in result.text
    assert result.text.count("Факт:") == 1
    assert "strip:inline_stance_lenin" in result.codes
    assert any(
        code in result.codes
        for code in ("strip:trailing_exact_fact_dup", "strip:trailing_triad_restart")
    )


def test_scrub_instruction_dump_keeps_lenin_prose() -> None:
    result = cleanup_answer_body(text=_load("stance_instruction_dump.in.txt"))
    assert "Запрещено комментировать" not in result.text
    assert "Запрещено выдумывать" not in result.text
    assert "Ленин (core_approval)" not in result.text
    assert "Ленин подчёркивал" in result.text
    assert "strip:instruction_dump" in result.codes


def test_normalize_star_junk_labels() -> None:
    result = cleanup_answer_body(text=_load("star_junk_labels.in.txt"))
    assert "Факт:" in result.text
    assert "Механизм:" in result.text
    assert "Вывод:" in result.text
    assert "\n*\n" not in result.text
    assert "Факт: **" not in result.text


def test_trailing_triad_restart_cut() -> None:
    result = cleanup_answer_body(text=_load("trailing_fact_en_restart.in.txt"))
    assert "капиталistic" not in result.text
    assert "Механism" not in result.text
    assert result.text.count("Факт:") == 1


def test_ru_core_stance_labels_stripped() -> None:
    result = cleanup_answer_body(text=_load("stance_ru_core_labels.in.txt"))
    assert "согласие (core_approval)" not in result.text
    assert "критика (core_criticism)" not in result.text
    assert "трудящихся" in result.text


def test_yellow_and_disclaimer_preserved() -> None:
    raw = _load("clean_with_yellow_disclaimer.in.txt")
    result = cleanup_answer_body(text=raw)
    assert "Ограниченный режим анализа" in result.text
    assert "Ответ сгенерирован ИИ" in result.text
    assert "Корнев" in result.text


def test_idempotent_second_pass() -> None:
    raw = _load("stance_instruction_dump.in.txt")
    first = cleanup_answer_body(text=raw)
    second = cleanup_answer_body(text=first.text)
    assert second.text == first.text


def test_soft_integrity_does_not_hard_fail() -> None:
    cfg = QualityPostcheckConfig(integrity_enforce_mode="soft")
    text = "Факт: заявление о может быть ложным.\nМеханизм: анализ.\nВывод: итог."
    result = cleanup_answer_body(text=text, config=cfg)
    assert result.metadata["integrity_error"] is True
    assert result.metadata["postprocess_hard_fail"] is False
    assert is_publishable_analysis(
        text=result.text,
        metadata={"structure_error": False, **result.metadata},
    )


def test_strict_integrity_sets_hard_fail() -> None:
    cfg = QualityPostcheckConfig(integrity_enforce_mode="strict")
    text = "Факт: заявление о может быть ложным.\nМеханизм: анализ.\nВывод: итог."
    result = cleanup_answer_body(text=text, config=cfg)
    assert result.metadata["postprocess_hard_fail"] is True
    assert not is_publishable_analysis(
        text=result.text,
        metadata={"structure_error": False, **result.metadata},
    )


def test_artifact_pass_wires_body_cleanup() -> None:
    cfg = QualityPostcheckConfig()
    body = _load("stance_trailing_fact_bsw.in.txt")
    result = apply_artifact_pass(text=body, config=cfg)
    assert "Ленин (agreement)" not in result.text
    assert result.text.count("Факт:") == 1
    assert result.metadata.get("body_cleanup_codes")


def test_qa2229_broken_core_lenin_stance() -> None:
    result = cleanup_answer_body(text=_load("core_lenin_broken_stance.in.txt"))
    assert "core_" not in result.text
    assert "Ленин (core_" not in result.text
    assert "strip:inline_stance_lenin" in result.codes
    assert "integrity:residual_stance" not in result.metadata["integrity_codes"]


def test_qa2229_md_hash_dash_tail() -> None:
    result = cleanup_answer_body(text=_load("md_hash_dash_tail.in.txt"))
    assert "--- ##" not in result.text
    assert not result.text.rstrip().endswith("##.")
    assert any(
        code in result.codes
        for code in ("strip:md_debris_cluster", "strip:terminal_md_debris")
    )
    assert "integrity:md_debris" not in result.metadata["integrity_codes"]


def test_qa2229_prompt_task_tail() -> None:
    result = cleanup_answer_body(text=_load("prompt_task_tail.in.txt"))
    assert "Задача" not in result.text
    assert "краткий анализ" not in result.text.casefold()
    assert "strip:prompt_task_tail" in result.codes
    assert "integrity:prompt_task_echo" not in result.metadata["integrity_codes"]


def test_qa2229_bold_section_labels() -> None:
    result = cleanup_answer_body(text=_load("bold_section_labels.in.txt"))
    assert "**Механизм:**" not in result.text
    assert "**Вывод:**" not in result.text
    assert "Механизм:" in result.text
    assert "Вывод:" in result.text
    assert "fix:inline_bold_label" in result.codes


def test_negative_legitimate_core_prose_and_slash() -> None:
    text = (
        "Факт: доклад о core_issue экономики.\n"
        "Механизм: ссылка https://example.com/path/критикой.\n"
        "Вывод: Ленин подчёркивал роль партии."
    )
    result = cleanup_answer_body(text=text)
    assert "core_issue" in result.text
    assert "https://example.com/path/критикой" in result.text
    assert "Ленин подчёркивал" in result.text
    assert "strip:inline_stance_lenin" not in result.codes
    assert "strip:prompt_task_tail" not in result.codes


def test_negative_single_md_separator_not_global_cut() -> None:
    text = "Факт: событие.\nМеханизм: анализ --- детали класса.\nВывод: итог."
    result = cleanup_answer_body(text=text)
    assert "анализ --- детали" in result.text
    assert "strip:terminal_md_debris" not in result.codes
    assert "strip:md_debris_cluster" not in result.codes


def test_inline_label_spacing_after_sentence_flatten() -> None:
    """Live QA failure: finalize joins sentences; labels stay inline with ' : '."""
    flat = (
        "Факт: событие произошло. Механизм : класс давит на трудящихся. "
        "Вывод : нужен контроль масс."
    )
    result = cleanup_answer_body(text=flat)
    assert "Механизм :" not in result.text
    assert "Вывод :" not in result.text
    assert "Механизм:" in result.text
    assert "Вывод:" in result.text
    assert "fix:label_spacing" in result.codes


def test_inline_trailing_triad_restart_after_flatten() -> None:
    flat = (
        "Факт: первое. Механизм: анализ. Вывод: итог. "
        "--- [empty] --- Факт: первое. Механизм: мусор."
    )
    result = cleanup_answer_body(text=flat)
    assert result.text.count("Факт:") == 1
    assert "--- [empty] ---" not in result.text
    assert any(
        code in result.codes
        for code in ("strip:trailing_exact_fact_dup", "strip:trailing_triad_restart")
    )


def test_finalize_then_cleanup_preserves_triad_breaks() -> None:
    from src.core.generation.text_postprocess import finalize_generated_text

    raw = (
        "Факт: событие произошло.\n"
        "Механизм : класс давит на трудящихся.\n"
        "Вывод : нужен контроль масс."
    )
    finalized, _meta = finalize_generated_text(raw)
    assert "\nМеханизм" in finalized or finalized.count("\n") >= 2
    result = cleanup_answer_body(text=finalized)
    assert "Механизм:" in result.text
    assert "Вывод:" in result.text
    assert "Механизм :" not in result.text


def test_disclaimer_glued_to_vyvod_still_normalizes_labels() -> None:
    text = (
        "Факт: событие.\n"
        "Механизм : анализ класса.\n"
        "Вывод : итог для масс. "
        "Ответ сгенерирован ИИ в образовательных целях "
        "(симуляция на основе трудов В.И. Ленина) и не является призывом к действию."
    )
    result = cleanup_answer_body(text=text)
    assert "Вывод :" not in result.text
    assert "Вывод:" in result.text
    assert "Ответ сгенерирован ИИ" in result.text
    assert "fix:label_spacing" in result.codes


def test_integrity_residual_codes_for_unscrubbed_debris() -> None:
    from src.core.generation.answer_body_cleanup import detect_integrity_issues

    text = (
        "Вывод: итог. — Ленин (core_ Lenin ). --- ## --- ##. "
        "Задача: краткий анализ в стиле Ленина. «[место]»"
    )
    codes = detect_integrity_issues(text)
    assert "integrity:residual_stance" in codes
    assert "integrity:md_debris" in codes
    assert "integrity:prompt_task_echo" in codes
    assert "integrity:mesto_marker" in codes
