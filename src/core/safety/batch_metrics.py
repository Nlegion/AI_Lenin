"""Batch routing / quote quality metrics helpers."""

from __future__ import annotations

import re
from typing import Any

from src.core.generation.quote_postcheck import check_critical_attribution
from src.core.generation.text_normalize import normalize_for_grounding

_ANSWER_QUOTE = re.compile(
    r"«([^»]{3,400})»|\"([^\"]{3,400})\"|„([^“]{3,400})“|“([^”]{3,400})”"
)
_HAS_ATTR = re.compile(r"(?i)\bтом\s*\d+|\bстр\.?\s*\d+|\[source:|/pss/|\\pss\\")


def routing_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = max(len(rows), 1)
    deny = sum(
        1
        for r in rows
        if r.get("skipped_llm_reason") == "pre_deny"
        or r.get("decision") in {"deny", "hard_block"}
    )
    review = sum(
        1
        for r in rows
        if r.get("skipped_llm_reason") == "pre_quarantine"
        or r.get("decision") in {"quarantine", "review"}
    )
    skip = sum(
        1
        for r in rows
        if r.get("skipped_llm_reason") == "out_of_scope_skip" or r.get("decision") == "skip"
    )
    yellow = sum(1 for r in rows if r.get("risk_tier") == "yellow")
    allow = sum(1 for r in rows if not r.get("blocked") and r.get("status") != "blocked")
    lengths = [len(str(r.get("answer") or "")) for r in rows if r.get("answer")]
    redact = sum(1 for r in rows if "[обезличено]" in str(r.get("answer") or ""))
    return {
        "hard_deny_rate": deny / total,
        "deny_rate": deny / total,
        "review_rate": review / total,
        "soft_skip_rate": skip / total,
        "skip_rate": skip / total,
        "yellow_rate": yellow / total,
        "allow_rate": allow / total,
        "mean_answer_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "redact_artifact_rate": redact / total,
        "n": float(len(rows)),
    }


def quote_grounding_rates(*, answers_and_contexts: list[tuple[str, str]]) -> dict[str, float]:
    """quoted span in answer ⊆ normalized context → grounded."""
    grounded = 0
    halluc = 0
    quoted = 0
    for answer, context in answers_and_contexts:
        ctx_norm = normalize_for_grounding(context)
        spans = []
        for match in _ANSWER_QUOTE.finditer(answer or ""):
            span = next((g for g in match.groups() if g), None)
            if span:
                spans.append(span.strip())
        for span in spans:
            quoted += 1
            if normalize_for_grounding(span) in ctx_norm:
                grounded += 1
            else:
                halluc += 1
    denom = max(quoted, 1)
    return {
        "quote_span_grounding_rate": grounded / denom if quoted else 1.0,
        "hallucinated_quote_rate": halluc / denom if quoted else 0.0,
        "quoted_spans": float(quoted),
        "quote_usage_rate": (1.0 if quoted else 0.0) if len(answers_and_contexts) == 1 else (
            sum(1 for a, _ in answers_and_contexts if _ANSWER_QUOTE.search(a or "")) / max(len(answers_and_contexts), 1)
        ),
    }


def critical_attribution_rates(*, answers: list[str]) -> dict[str, float]:
    """Denominator = answers that contain attribution markers."""
    with_attr = [a for a in answers if a and _HAS_ATTR.search(a)]
    if not with_attr:
        return {
            "critical_attribution_hallucination_rate": 0.0,
            "attribution_answers": 0.0,
        }
    bad = 0
    for answer in with_attr:
        codes = check_critical_attribution(answer, candidates=[])
        if codes:
            bad += 1
    return {
        "critical_attribution_hallucination_rate": bad / len(with_attr),
        "attribution_answers": float(len(with_attr)),
    }


def path_leak_rate(*, answers: list[str]) -> float:
    if not answers:
        return 0.0
    leaks = sum(1 for a in answers if a and ("/pss/" in a or "[source:" in a.lower() or "\\pss\\" in a))
    return leaks / len(answers)


def loop_rates(*, rows: list[dict[str, Any]]) -> dict[str, float]:
    total = max(len(rows), 1)
    loops = sum(1 for r in rows if r.get("paragraph_loop_detected") or r.get("metadata", {}).get("paragraph_loop_detected"))
    return {"paragraph_loop_rate": loops / total}


_STATIC_SAFE = "Не удалось сформировать корректный анализ по данной новости."
_STATIC_INSUFFICIENT = "Недостаточно данных для анализа."
_FALLBACK_HINTS = (
    _STATIC_SAFE,
    _STATIC_INSUFFICIENT,
    "не содержит достаточного социально-экономического контекста",
)


def template_fallback_rates(*, answers: list[str]) -> dict[str, float]:
    """Share of LLM answers that collapsed to static/fallback templates."""
    total = max(len(answers), 1)
    safe_n = 0
    insuff_n = 0
    any_n = 0
    lengths: list[int] = []
    for answer in answers:
        text = (answer or "").strip()
        lengths.append(len(text))
        is_safe = _STATIC_SAFE in text
        is_insuff = _STATIC_INSUFFICIENT in text
        is_fb = any(hint in text for hint in _FALLBACK_HINTS)
        if is_safe:
            safe_n += 1
        if is_insuff:
            insuff_n += 1
        if is_fb:
            any_n += 1
    return {
        "static_safe_template_share": safe_n / total,
        "static_insufficient_template_share": insuff_n / total,
        "template_fallback_share": any_n / total,
        "avg_answer_chars": (sum(lengths) / len(lengths)) if lengths else 0.0,
    }


_REASONING_CONNECTORS = (
    "потому что",
    "поэтому",
    "следовательно",
    "таким образом",
    "в результате",
    "поскольку",
)


def depth_quality_proxies(*, answers: list[str], news_blobs: list[str] | None = None) -> dict[str, float]:
    """Automated depth proxies for Stage 3 (calibrate before using as hard gates)."""
    total = max(len(answers), 1)
    connector_hits = 0
    diversity_sum = 0.0
    template_hits = 0
    fact_anchor_hits = 0
    news_blobs = news_blobs or [""] * len(answers)
    for idx, answer in enumerate(answers):
        text = (answer or "").lower()
        if any(c in text for c in _REASONING_CONNECTORS):
            connector_hits += 1
        tokens = re.findall(r"[а-яёa-z0-9]{3,}", text, flags=re.IGNORECASE)
        if tokens:
            diversity_sum += len(set(tokens)) / len(tokens)
        if any(h in (answer or "") for h in _FALLBACK_HINTS):
            template_hits += 1
        news = (news_blobs[idx] if idx < len(news_blobs) else "") or ""
        news_nums = set(re.findall(r"\d+", news))
        ans_nums = set(re.findall(r"\d+", answer or ""))
        news_caps = set(re.findall(r"\b[А-ЯЁ][а-яё]{3,}\b", news))
        ans_caps = set(re.findall(r"\b[А-ЯЁ][а-яё]{3,}\b", answer or ""))
        if (news_nums and news_nums & ans_nums) or (news_caps and news_caps & ans_caps):
            fact_anchor_hits += 1
    return {
        "reasoning_connector_rate": connector_hits / total,
        "lexical_diversity": diversity_sum / total,
        "template_phrase_rate": template_hits / total,
        "fact_anchor_rate": fact_anchor_hits / total,
    }


def stage0_template_rollback_signal(
    *,
    current_share: float,
    baseline_share: float,
    deny_rate: float | None = None,
    baseline_deny_rate: float | None = None,
    absolute_cap: float = 0.05,
    relative_improve: float = 0.50,
    min_absolute_gain_pp: float = 0.02,
) -> dict[str, Any]:
    """Combined absolute/relative Stage 0 success check (plan rollback guardrails)."""
    gain = baseline_share - current_share
    relative_ok = (
        baseline_share > 1e-9
        and (gain / baseline_share) >= relative_improve
        and gain >= min_absolute_gain_pp
    )
    absolute_ok = current_share < absolute_cap
    success = absolute_ok or relative_ok
    deny_spike = False
    if deny_rate is not None and baseline_deny_rate is not None:
        deny_spike = (deny_rate - baseline_deny_rate) > 0.02
    return {
        "success": bool(success and not deny_spike),
        "absolute_ok": absolute_ok,
        "relative_ok": relative_ok,
        "deny_spike": deny_spike,
        "current_share": current_share,
        "baseline_share": baseline_share,
        "action": "ok" if success and not deny_spike else "rollback_review",
    }


def drift_vs_baseline(
    current: dict[str, float],
    baseline: dict[str, float],
    *,
    keys: tuple[str, ...] = (
        "deny_rate",
        "review_rate",
        "skip_rate",
        "allow_rate",
        "redact_artifact_rate",
        "hard_deny_rate",
        "soft_skip_rate",
    ),
) -> dict[str, Any]:
    """Flag relative drift >20% on rate keys."""
    flags: list[str] = []
    details: dict[str, float] = {}
    for key in keys:
        if key not in current and key not in baseline:
            continue
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
        "manual_review_required": severity != "ok",
        "sla_hint": (
            "≤15min manual rollback (primary=maintainer, backup=architect)"
            if severity == "critical_gt_40"
            else "≤1h review" if severity == "warn_20_40" else "none"
        ),
    }
