"""Unit tests for quote_mode selection and overlap."""

from __future__ import annotations

from src.core.generation.quote_mode import (
    answer_has_quotes,
    lexical_overlap,
    select_quote_mode,
    strip_quotes,
)


def test_lexical_overlap_positive() -> None:
    news = "Инфляция и безработица выросли в экономике страны"
    chunk = "Рост инфляции и безработицы отражает кризис экономики"
    assert lexical_overlap(news=news, chunk=chunk) >= 0.15


def test_select_quote_mode_requires_overlap_and_quotes() -> None:
    news = "Инфляция выросла до рекордных значений"
    chunks = [
        ("1", 0.9, "«Капитализм порождает кризисы» — инфляция и безработица"),
        ("2", 0.5, "Общая фраза без связи"),
    ]
    mode, _ = select_quote_mode(news=news, chunks=chunks, overlap_threshold=0.15)
    assert mode == "quote"


def test_select_principles_when_no_quote() -> None:
    news = "Инфляция выросла"
    chunks = [("1", 0.9, "Инфляция и цены без кавычек")]
    mode, _ = select_quote_mode(news=news, chunks=chunks)
    assert mode == "principles"


def test_strip_quotes_postcheck() -> None:
    text = "Как писал Ленин: «выдуманная цитата»."
    assert answer_has_quotes(text)
    cleaned = strip_quotes(text)
    assert not answer_has_quotes(cleaned)
