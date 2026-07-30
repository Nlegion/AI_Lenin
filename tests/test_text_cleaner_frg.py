"""Regression: TextCleaner must not turn possessive её into ФРГ."""

from __future__ import annotations

from src.core.text_cleaner import TextCleaner


def test_clean_text_preserves_ee_possessive() -> None:
    cleaner = TextCleaner()
    sample = "обеспечения её достойной жизни при призме её воздействия"
    result = cleaner.clean_text(sample)
    assert "ФРГ" not in result


def test_clean_text_does_not_map_yo_to_frg() -> None:
    cleaner = TextCleaner()
    # Even if spelling path yielded "Рё", political map must not rewrite to FRG.
    result = cleaner.clean_text("анализ Рё влияния капитала")
    assert "ФРГ" not in result


def test_clean_text_still_fixes_long_typos() -> None:
    cleaner = TextCleaner()
    result = cleaner.clean_text("капиталистическихого способа")
    assert "капиталистического" in result
