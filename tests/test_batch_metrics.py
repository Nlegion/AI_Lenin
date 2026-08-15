"""Tests for batch routing metrics and drift SLA hints."""

from __future__ import annotations

from src.core.safety.batch_metrics import drift_vs_baseline, routing_rates


def test_routing_rates_and_drift() -> None:
    rows = [
        {"blocked": False, "status": "ok", "answer": "abc"},
        {"blocked": True, "skipped_llm_reason": "pre_deny", "answer": "x"},
        {"blocked": True, "skipped_llm_reason": "out_of_scope_skip", "answer": "y"},
    ]
    rates = routing_rates(rows)
    assert rates["n"] == 3
    baseline = {
        "deny_rate": 0.1,
        "skip_rate": 0.1,
        "allow_rate": 0.8,
        "redact_artifact_rate": 0.0,
    }
    drift = drift_vs_baseline(rates, baseline)
    assert drift["severity"] in {"warn_20_40", "critical_gt_40", "ok"}
