"""Batch routing / quote quality metrics helpers."""

from __future__ import annotations

from typing import Any


def routing_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = max(len(rows), 1)
    deny = sum(1 for r in rows if r.get("skipped_llm_reason") == "pre_deny" or r.get("decision") == "deny")
    skip = sum(
        1
        for r in rows
        if r.get("skipped_llm_reason") == "out_of_scope_skip" or r.get("decision") == "skip"
    )
    allow = sum(1 for r in rows if not r.get("blocked") and r.get("status") != "blocked")
    lengths = [len(str(r.get("answer") or "")) for r in rows if r.get("answer")]
    redact = sum(1 for r in rows if "[обезличено]" in str(r.get("answer") or ""))
    return {
        "deny_rate": deny / total,
        "skip_rate": skip / total,
        "allow_rate": allow / total,
        "mean_answer_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "redact_artifact_rate": redact / total,
        "n": float(len(rows)),
    }


def quote_grounding_rates(*, answers_and_contexts: list[tuple[str, str]]) -> dict[str, float]:
    """quoted span in answer ⊆ context → grounded; quoted not in context → halluc."""
    import re

    grounded = 0
    halluc = 0
    quoted = 0
    for answer, context in answers_and_contexts:
        spans = re.findall(r"[«\"]([^»\"]{3,120})[»\"]", answer)
        for span in spans:
            quoted += 1
            if span.lower() in context.lower():
                grounded += 1
            else:
                halluc += 1
    denom = max(quoted, 1)
    return {
        "quote_span_grounding_rate": grounded / denom if quoted else 1.0,
        "hallucinated_quote_rate": halluc / denom if quoted else 0.0,
        "quoted_spans": float(quoted),
    }


def drift_vs_baseline(
    current: dict[str, float],
    baseline: dict[str, float],
    *,
    keys: tuple[str, ...] = ("deny_rate", "skip_rate", "allow_rate", "redact_artifact_rate"),
) -> dict[str, Any]:
    """Flag relative drift >20% on rate keys."""
    flags: list[str] = []
    details: dict[str, float] = {}
    for key in keys:
        base = float(baseline.get(key, 0.0))
        cur = float(current.get(key, 0.0))
        if base <= 1e-9:
            rel = 1.0 if cur > 1e-9 else 0.0
        else:
            rel = abs(cur - base) / base
        details[key] = rel
        if rel > 0.20:
            flags.append(key)
    severity = "ok"
    max_rel = max(details.values()) if details else 0.0
    if max_rel > 0.40:
        severity = "critical_gt_40"
    elif max_rel > 0.20:
        severity = "warn_20_40"
    return {
        "severity": severity,
        "flagged_keys": flags,
        "relative_drift": details,
        "sla_hint": (
            "≤15min manual rollback (primary=maintainer, backup=architect)"
            if severity == "critical_gt_40"
            else "≤1h review" if severity == "warn_20_40" else "none"
        ),
    }
