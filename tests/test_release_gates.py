"""Unit tests for release_gates metric tolerance helper."""

from __future__ import annotations

from src.core.settings.release_gates import load_release_gates, metric_passes


def test_metric_passes_higher_with_tolerance() -> None:
    assert metric_passes(
        value=0.83,
        threshold=0.85,
        direction="higher",
        tolerance_relative=0.03,
    )
    assert not metric_passes(
        value=0.80,
        threshold=0.85,
        direction="higher",
        tolerance_relative=0.03,
    )


def test_metric_passes_lower_with_tolerance() -> None:
    assert metric_passes(
        value=0.051,
        threshold=0.05,
        direction="lower",
        tolerance_relative=0.03,
    )
    assert not metric_passes(
        value=0.06,
        threshold=0.05,
        direction="lower",
        tolerance_relative=0.03,
    )


def test_load_release_gates_has_metrics() -> None:
    gates = load_release_gates()
    assert gates.version
    assert "recall_at_5" in gates.rag_quality.metrics
    assert gates.rag_quality.metrics["recall_at_5"].direction == "higher"
