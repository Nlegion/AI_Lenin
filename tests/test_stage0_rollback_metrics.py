"""Stage 0 rollback signal and depth proxy smoke tests."""

from __future__ import annotations

from src.core.safety.batch_metrics import (
    depth_quality_proxies,
    stage0_template_rollback_signal,
    template_fallback_rates,
)


def test_template_fallback_rates() -> None:
    answers = [
        "Развернутый анализ инфляции и тарифов.",
        "Не удалось сформировать корректный анализ по данной новости.",
        "Недостаточно данных для анализа.",
    ]
    stats = template_fallback_rates(answers=answers)
    assert abs(stats["static_safe_template_share"] - 1 / 3) < 1e-9
    assert abs(stats["template_fallback_share"] - 2 / 3) < 1e-9


def test_stage0_rollback_combined_absolute_relative() -> None:
    # Low baseline: relative 50% may be impossible; absolute cap still succeeds.
    low = stage0_template_rollback_signal(current_share=0.03, baseline_share=0.04)
    assert low["success"] is True
    assert low["absolute_ok"] is True

    # High baseline with strong relative gain.
    improved = stage0_template_rollback_signal(current_share=0.10, baseline_share=0.30)
    assert improved["success"] is True
    assert improved["relative_ok"] is True

    # Deny spike forces rollback review.
    spiked = stage0_template_rollback_signal(
        current_share=0.02,
        baseline_share=0.30,
        deny_rate=0.20,
        baseline_deny_rate=0.16,
    )
    assert spiked["deny_spike"] is True
    assert spiked["success"] is False


def test_depth_quality_proxies() -> None:
    stats = depth_quality_proxies(
        answers=["Рост тарифов потому что монополии усиливают давление."],
        news_blobs=["Рост тарифов ЖКХ в регионах"],
    )
    assert stats["reasoning_connector_rate"] == 1.0
    assert stats["fact_anchor_rate"] == 1.0
