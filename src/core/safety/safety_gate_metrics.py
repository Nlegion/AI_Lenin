"""Observability helpers for SafetyGate dual-run and quality recovery."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.core.safety.safety_gate_types import ShadowCompareResult


def shadow_compare_to_log_fields(
    *,
    compare: ShadowCompareResult,
    item_id: str | None = None,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "old_decision": compare.old_decision.decision if compare.old_decision else None,
        "new_decision": compare.new_decision.decision if compare.new_decision else None,
        "enforced_decision": compare.enforced.decision,
        "decision_match": compare.decision_match,
        "reason_diff": list(compare.reason_diff),
        "config_version_hash": compare.config_version_hash,
        "old_codes": list(compare.old_decision.reason_codes)
        if compare.old_decision
        else [],
        "new_codes": list(compare.new_decision.reason_codes)
        if compare.new_decision
        else [],
        "risk_tier": compare.enforced.risk_tier,
        "safety_gate_latency_ms": compare.enforced.latency_ms,
    }


def aggregate_gate_shares(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = max(len(rows), 1)
    decisions = Counter(
        str(r.get("decision") or r.get("enforced_decision") or "") for r in rows
    )
    tiers = Counter(str(r.get("risk_tier") or "") for r in rows)
    deny_like = decisions.get("deny", 0) + decisions.get("hard_block", 0)
    review_like = decisions.get("quarantine", 0) + decisions.get("review", 0)
    return {
        "gate_allow_share": decisions.get("allow", 0) / total,
        "gate_deny_share": deny_like / total,
        "gate_review_share": review_like / total,
        "gate_skip_share": decisions.get("skip", 0) / total,
        "gate_quarantine_share": review_like / total,
        "gate_yellow_share": tiers.get("yellow", 0) / total,
        "n": float(len(rows)),
    }


def alert_levels(
    *,
    mismatch_rate: float,
    red_allow_leak_rate: float,
    yellow_share_delta: float,
    mean_output_chars_delta: float,
    template_share: float,
    deny_rate_delta: float,
) -> dict[str, Any]:
    """Multi-level alerts: warning vs critical."""
    warnings: list[str] = []
    critical: list[str] = []
    if mismatch_rate >= 0.05:
        warnings.append("parity_drift")
    if mismatch_rate >= 0.15:
        critical.append("parity_drift_severe")
    if abs(yellow_share_delta) >= 0.05:
        warnings.append("yellow_share_drift")
    if mean_output_chars_delta <= -0.20:
        warnings.append("output_length_drop")
    if red_allow_leak_rate > 0:
        critical.append("red_allow_leakage")
    if deny_rate_delta > 0.05:
        critical.append("deny_spike")
    if template_share >= 0.20:
        critical.append("template_spike")
    elif template_share >= 0.10:
        warnings.append("template_elevated")
    level = "ok"
    if critical:
        level = "critical"
    elif warnings:
        level = "warning"
    return {"level": level, "warnings": warnings, "critical": critical}
