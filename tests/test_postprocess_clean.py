"""postprocess_clean contract, phases, fixtures, and post-guard invariant."""

from __future__ import annotations

from pathlib import Path

from src.core.generation.postprocess_clean import (
    PostProcessInput,
    apply_terminal_public_scrub,
    map_postprocess_status,
    resolve_clean_mode,
    run_postprocess,
    scrub_after_output_guard,
)
from src.core.generation.publishability import is_publishable_analysis
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "answer_postprocess"
FORBIDDEN = (
    "Ленин (core_",
    "core_approval",
    "<|im_start|>",
    "[multi-stance]",
    "«[место]»",
    "[обезличено]",
    "Задача: краткий анализ",
)


def _pre(text: str, **kwargs) -> object:
    cfg = kwargs.pop("config", QualityPostcheckConfig())
    return run_postprocess(
        PostProcessInput(raw_text=text, phase="pre_guard", config=cfg, **kwargs)
    )


def test_status_mapping_does_not_override_guard() -> None:
    assert (
        map_postprocess_status(postprocess_hard_fail=True, structure_error=False)
        == "blocked"
    )
    assert (
        map_postprocess_status(postprocess_hard_fail=False, structure_error=True)
        == "needs_review"
    )
    assert (
        map_postprocess_status(postprocess_hard_fail=False, structure_error=False)
        == "clean"
    )


def test_default_mode_is_live() -> None:
    assert resolve_clean_mode(QualityPostcheckConfig()) == "live"
    assert (
        resolve_clean_mode(QualityPostcheckConfig(postprocess_clean_mode="shadow"))
        == "shadow"
    )


def test_pre_guard_fixture_idempotent() -> None:
    raw = (FIXTURES / "stance_instruction_dump.in.txt").read_text(encoding="utf-8")
    first = _pre(raw)
    second = _pre(first.cleaned_text)
    assert second.cleaned_text == first.cleaned_text
    assert "Ленин (core_approval)" not in first.cleaned_text
    assert "Ленин подчёркивал" in first.cleaned_text


def test_pre_guard_preserves_safety_tails() -> None:
    raw = (FIXTURES / "clean_with_yellow_disclaimer.in.txt").read_text(encoding="utf-8")
    result = _pre(raw)
    assert "Ограниченный режим анализа" in result.cleaned_text
    assert "Ответ сгенерирован ИИ" in result.cleaned_text


def test_post_guard_strips_newsguard_mesto() -> None:
    body = _pre("Факт: Иванов сообщил. Механизм: анализ. Вывод: итог.")
    guarded = body.cleaned_text.replace("Иванов", "«[место]»")
    result = run_postprocess(PostProcessInput(raw_text=guarded, phase="post_guard"))
    assert "«[место]»" not in result.cleaned_text
    assert "strip:mesto_marker" in result.codes
    again = run_postprocess(
        PostProcessInput(raw_text=result.cleaned_text, phase="post_guard")
    )
    assert again.cleaned_text == result.cleaned_text


def test_soft_integrity_stays_publishable() -> None:
    cfg = QualityPostcheckConfig(integrity_enforce_mode="soft")
    text = "Факт: заявление о может быть ложным.\nМеханизм: анализ.\nВывод: итог."
    result = _pre(text, config=cfg)
    assert result.status != "blocked"
    assert result.postprocess_hard_fail is False
    assert is_publishable_analysis(
        text=result.cleaned_text,
        metadata=result.to_legacy_metadata(),
    )


def test_strict_integrity_blocks_publish() -> None:
    cfg = QualityPostcheckConfig(integrity_enforce_mode="strict")
    text = "Факт: заявление о может быть ложным.\nМеханизм: анализ.\nВывод: итог."
    result = _pre(text, config=cfg)
    assert result.status == "blocked"
    assert result.postprocess_hard_fail is True
    assert not is_publishable_analysis(
        text=result.cleaned_text,
        metadata=result.to_legacy_metadata(),
    )


def test_skip_structure_enforce_does_not_force_review() -> None:
    result = _pre("короткий текст без секций", skip_structure_enforce=True)
    assert result.structure_error is False
    assert result.status == "clean"


def test_persist_helper_scrubs_after_guard_insert() -> None:
    text = "Факт: чиновник из «[место]» заявил. Механизм: анализ. Вывод: итог."
    cleaned = scrub_after_output_guard(text)
    assert "«[место]»" not in cleaned
    assert "чиновник из" in cleaned


def test_terminal_meta_codes_written() -> None:
    meta: dict = {}
    apply_terminal_public_scrub(
        "Факт: x [multi-stance] Механизм: y. Вывод: z.",
        quality_meta=meta,
    )
    assert meta.get("final_public_scrub_codes")
    assert meta.get("postprocess_status_post_guard") == "clean"


def test_fixture_corpus_has_no_forbidden_public_markers() -> None:
    for path in sorted(FIXTURES.glob("*.in.txt")):
        result = _pre(path.read_text(encoding="utf-8"))
        public = run_postprocess(
            PostProcessInput(raw_text=result.cleaned_text, phase="post_guard")
        )
        for token in FORBIDDEN:
            assert token not in public.cleaned_text, f"{path.name} leaked {token}"
