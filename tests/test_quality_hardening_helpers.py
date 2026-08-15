"""Unit tests for text postprocess and context budget helpers."""

from __future__ import annotations

from src.core.generation.context_budget import (
    BudgetState,
    approx_tokens,
    clip_context_by_chunks,
    shrink_budget,
)
from src.core.generation.prompt_adapter import (
    FALLBACK_LEGACY_MARKER,
    build_chat_request,
)
from src.core.generation.quality_hooks import (
    _enforce_required_structure,
    _has_required_structure,
)
from src.core.generation.text_postprocess import (
    clamp_answer_length,
    dedupe_consecutive_sentences,
    finalize_generated_text,
    strip_truncation_markers,
)
from src.core.safety.groundedness_warn import news_groundedness


def test_strip_truncation_markers() -> None:
    assert "[truncated]" not in strip_truncation_markers("text\n...[truncated]")


def test_dedupe_consecutive_sentences() -> None:
    text = "Тезис один. Тезис один. Тезис два."
    cleaned, meta = dedupe_consecutive_sentences(text)
    assert meta["consecutive_repeat_removed"] == 1
    assert cleaned.count("Тезис один") == 1


def test_finalize_strips_and_trims() -> None:
    text, meta = finalize_generated_text("Первое. Первое. Обрыв без точки")
    assert meta["consecutive_repeat_removed"] == 1
    assert text.endswith(".")


def test_finalize_restores_triad_section_breaks() -> None:
    flat = (
        "Факт: первое предложение. Механизм: второе предложение. "
        "Вывод: третье предложение."
    )
    text, _meta = finalize_generated_text(flat)
    assert "\nМеханизм:" in text
    assert "\nВывод:" in text


def test_clamp_answer_length_trims_very_long_text() -> None:
    long_text = ("Факт. " * 400).strip()
    clamped, changed = clamp_answer_length(long_text, max_chars=500)
    assert changed is True
    assert len(clamped) <= 500


def test_shrink_budget_chunks_first() -> None:
    state = BudgetState(max_context_chars=5500, max_context_chunks=7)
    assert shrink_budget(state)
    assert state.max_context_chunks == 5
    assert state.shrink_steps[0]["context_shrink_step"] == "chunks"


def test_clip_context_by_chunks() -> None:
    ctx = "a\n\nb\n\nc\n\nd"
    assert clip_context_by_chunks(ctx, max_chunks=2) == "a\n\nb"


def test_approx_tokens_positive() -> None:
    assert approx_tokens("abcdefgh") >= 2


def test_legacy_fallback_marker_in_chat_request() -> None:
    req = build_chat_request(
        news_title="t",
        news_content="c",
        context="quote",
        max_context_chars=1000,
        legacy_fallback=True,
    )
    assert FALLBACK_LEGACY_MARKER in req.user_content


def test_news_groundedness_keyterm() -> None:
    result = news_groundedness(
        analysis="Инфляция в еврозоне отражает кризис.",
        news_title="Рост инфляции в еврозоне",
        news_content="Потребительская инфляция выросла.",
    )
    assert result.grounded
    assert result.matched_keyterms


def test_has_required_structure_accepts_spaced_and_bold_labels() -> None:
    text = (
        "**Факт:** В новости указан конкретный шаг.\n"
        "Механизм : Связь с теоретическим принципом.\n"
        "Вывод: Практическое следствие для оценки события."
    )
    assert _has_required_structure(text)


def test_enforce_required_structure_skips_long_nonstructured_text() -> None:
    long_text = "Это длинный анализ без явных меток. " * 80
    rebuilt, structure_ok, structure_error = _enforce_required_structure(long_text)
    assert rebuilt == long_text
    assert structure_ok is False
    assert structure_error is True
    assert "анализ опирается" not in rebuilt


def test_enforce_required_structure_never_injects_stub() -> None:
    text = "Короткий ответ без меток."
    rebuilt, structure_ok, structure_error = _enforce_required_structure(text)
    assert rebuilt == text
    assert structure_ok is False
    assert structure_error is True
    assert "Механизм: анализ опирается" not in rebuilt


def test_scaffold_preserves_fact_mechanism_conclusion_labels() -> None:
    from src.core.generation.output_artifacts import apply_artifact_pass
    from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

    structured = (
        "Факт: Правительство ввело регулирование.\n"
        "Механизм: Госкапитализм обслуживает монополии.\n"
        "Вывод: Нужен рабочий контроль."
    )
    result = apply_artifact_pass(text=structured, config=QualityPostcheckConfig())
    assert "Факт:" in result.text
    assert "Вывод:" in result.text
    assert "Механизм:" in result.text


def test_artifact_pass_strips_inline_context_tail() -> None:
    from src.core.generation.output_artifacts import apply_artifact_pass
    from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

    text = (
        "Факт: Рост цен.\n"
        "Механизм: Давление капитала.\n"
        "Вывод: Нужен контроль. --- контекст — Ленин (core_критика) "
        "контекст — Ленин (core_согласие) --- Добавить краткий комментарий. "
        "В стилизованной интерпретации: --- Доказательная база: [1] (intellectual/...) ---."
    )
    result = apply_artifact_pass(text=text, config=QualityPostcheckConfig())
    lowered = result.text.lower()
    assert "контекст — ленин" not in lowered
    assert "добавить краткий комментарий" not in lowered
    assert "в стилизованной интерпретации" not in lowered
    assert "доказательная база" not in lowered


def test_artifact_pass_repairs_ryo_without_artifact_flag() -> None:
    from src.core.generation.output_artifacts import apply_artifact_pass
    from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

    text = "Рост цен ведёт к снижению Рё реальных доходов."
    result = apply_artifact_pass(
        text=text, config=QualityPostcheckConfig(), item_id="ryo"
    )
    assert "её реальных доходов" in result.text
    assert "artifact:mojibake_ryo" not in result.codes
    assert "detect:encoding_artifact" not in result.codes


def test_max_final_answer_chars_exported() -> None:
    from src.core.generation.text_postprocess import MAX_FINAL_ANSWER_CHARS

    assert MAX_FINAL_ANSWER_CHARS == 1800
