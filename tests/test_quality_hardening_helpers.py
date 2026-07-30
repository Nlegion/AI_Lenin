"""Unit tests for text postprocess and context budget helpers."""

from __future__ import annotations

from src.core.generation.context_budget import (
    BudgetState,
    approx_tokens,
    clip_context_by_chunks,
    shrink_budget,
)
from src.core.generation.prompt_adapter import FALLBACK_LEGACY_MARKER, build_chat_request
from src.core.generation.text_postprocess import (
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
