"""Publisher path: footer length reserve and no bare redact artifacts."""

from __future__ import annotations

from src.core.publisher import (
    AI_DISCLAIMER,
    _extract_triad_sections,
    clean_telegram_text,
)


def test_ai_disclaimer_matches_research_sot() -> None:
    assert "исследовательских целях" in AI_DISCLAIMER
    assert "образовательных целях" not in AI_DISCLAIMER
    assert "призывом к действию" in AI_DISCLAIMER


def test_clean_telegram_preserves_footer_text() -> None:
    body = "Краткий анализ экономики.\n\n" + AI_DISCLAIMER
    cleaned = clean_telegram_text(body)
    assert AI_DISCLAIMER in cleaned
    assert "[обезличено]" not in cleaned


def test_extract_triad_from_flattened_single_line() -> None:
    body = (
        "Факт: В Домодедове произошёл пожар. "
        "Механизм: Экономия на безопасности. "
        "Вывод: Это симптом порядка."
    )
    sections = _extract_triad_sections(clean_telegram_text(body))
    assert sections.get("механизм", "").startswith("Экономия")
    assert sections.get("вывод", "").startswith("Это симптом")


def test_clean_telegram_keeps_triad_line_breaks() -> None:
    body = (
        "Факт: цены выросли.\n\n"
        "Механизм: капиталистическая конкуренция.\n\n"
        "Вывод: трудящиеся платят."
    )
    cleaned = clean_telegram_text(body)
    assert "\n" in cleaned
    sections = _extract_triad_sections(cleaned)
    assert sections.get("механизм") == "капиталистическая конкуренция."
    assert sections.get("вывод") == "трудящиеся платят."


def test_analysis_length_budget_keeps_short_footer() -> None:
    """Mirror publisher truncation policy: keep trailing short footer."""
    body = ("А" * 5000) + "\n\n" + AI_DISCLAIMER
    parts = body.rsplit("\n\n", 1)
    assert len(parts) == 2
    max_analysis = 3500
    budget = max(80, max_analysis - len(parts[1]) - 4)
    rebuilt = f"{parts[0][:budget].rstrip()}...\n\n{parts[1]}"
    assert AI_DISCLAIMER in rebuilt
    assert len(rebuilt) < len(body)
