"""Publisher path: footer length reserve and no bare redact artifacts."""

from __future__ import annotations

from src.core.publisher import clean_telegram_text


def test_clean_telegram_preserves_footer_text() -> None:
    footer = (
        "Ответ сгенерирован ИИ в образовательных целях "
        "(симуляция на основе трудов В.И. Ленина) и не является призывом к действию."
    )
    body = "Краткий анализ экономики.\n\n" + footer
    cleaned = clean_telegram_text(body)
    assert footer in cleaned
    assert "[обезличено]" not in cleaned


def test_analysis_length_budget_keeps_short_footer() -> None:
    """Mirror publisher truncation policy: keep trailing short footer."""
    footer = "Ответ сгенерирован ИИ в образовательных целях."
    body = ("А" * 5000) + "\n\n" + footer
    parts = body.rsplit("\n\n", 1)
    assert len(parts) == 2
    max_analysis = 3500
    budget = max(80, max_analysis - len(parts[1]) - 4)
    rebuilt = f"{parts[0][:budget].rstrip()}...\n\n{parts[1]}"
    assert footer in rebuilt
    assert len(rebuilt) < len(body)
