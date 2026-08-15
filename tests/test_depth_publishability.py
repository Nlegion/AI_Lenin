from __future__ import annotations

from src.core.generation.adaptive_context import adaptive_max_context_chars
from src.core.generation.publishability import (
    is_error_placeholder,
    is_publishable_analysis,
)


def test_adaptive_context_boosts_hard_topics() -> None:
    hard = adaptive_max_context_chars(
        primary="labor_economy",
        base_chars=3000,
        ctx_size=4096,
        max_tokens=512,
    )
    light = adaptive_max_context_chars(
        primary="sport",
        base_chars=3000,
        ctx_size=4096,
        max_tokens=512,
    )
    assert hard >= 3000
    assert light <= 3000
    assert hard > light


def test_publishability_blocks_structure_and_placeholders() -> None:
    assert is_error_placeholder("Ошибка анализа.")
    assert not is_publishable_analysis(
        text="Факт: x. Механизм: y. Вывод: z.",
        metadata={"structure_error": True},
    )
    assert not is_publishable_analysis(
        text="ok",
        metadata={"dialectical_outcome": "hold_review"},
    )
    assert is_publishable_analysis(
        text="Факт: x. Механизм: y. Вывод: z.",
        metadata={"structure_error": False, "dialectical_outcome": "ok"},
    )


def test_publishability_blocks_postprocess_hard_fail() -> None:
    assert not is_publishable_analysis(
        text="Факт: x. Механизм: y. Вывод: z.",
        metadata={"structure_error": False, "postprocess_hard_fail": True},
    )


def test_publishability_respects_refreshed_hold_outcome() -> None:
    # Simulates pipeline resync after post-QC flipped publish → hold_review.
    assert not is_publishable_analysis(
        text="Факт: x. Механизм: y. Вывод: z.",
        metadata={"dialectical_outcome": "hold_review", "structure_error": False},
    )
