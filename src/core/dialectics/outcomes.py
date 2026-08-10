"""Outcome helpers for DialecticalEngine (keeps engine.py under size limit)."""

from __future__ import annotations

import time
from typing import Any

from src.core.dialectics.config import DialecticalReasoningConfig
from src.core.dialectics.renderer import render_analysis_text
from src.core.dialectics.schemas import DialecticalResult, PrincipleCard, QualityReport


def finalize_result(
    *,
    result: DialecticalResult,
    reason_codes: list[str],
    timings: dict[str, float],
    started: float,
    has_r3: bool,
    config: DialecticalReasoningConfig,
) -> DialecticalResult:
    codes = list(dict.fromkeys([*reason_codes, *result.reason_codes]))
    if not has_r3 and "r3_absent" not in codes:
        codes.append("r3_absent")
    if result.quality.errors:
        codes.extend(result.quality.errors)
    result.reason_codes = list(dict.fromkeys(codes))
    rendered, truncated = render_analysis_text(result=result, config=config)
    result.rendered_text = rendered
    if truncated:
        result.reason_codes.append("length_clamped")
        result.metadata["render_truncated"] = True
    if result.quality.passed and "boilerplate_phrase" not in result.quality.errors:
        result.outcome = "publish"
    elif "insufficient_evidence" in result.reason_codes:
        result.outcome = "suppress"
    else:
        result.outcome = "hold_review"
    timings["total_ms"] = (time.perf_counter() - started) * 1000.0
    result.pass_timings_ms = timings
    return result


def simplified_result(
    *,
    fact: str,
    reason_codes: list[str],
    timings: dict[str, float],
    started: float,
    cards: list[PrincipleCard],
    config: DialecticalReasoningConfig,
) -> DialecticalResult:
    if "insufficient_evidence" in reason_codes and not cards:
        return terminal_result(
            outcome="suppress",
            reason_codes=reason_codes,
            timings=timings,
            started=started,
        )
    result = DialecticalResult(
        outcome="hold_review",
        reason_codes=list(reason_codes),
        fact=fact,
        mechanism_steps=[
            "Теоретическое обоснование ограничено доступным контекстом; "
            "полный диалектический разбор не применим."
        ],
        conclusion="Вывод ограничен: недостаточно оппозиции или применимости диалектики.",
        used_principles=list(cards[:1]),
        quality=QualityReport(passed=False, warnings=list(reason_codes)),
    )
    rendered, truncated = render_analysis_text(result=result, config=config)
    result.rendered_text = rendered
    if truncated:
        result.reason_codes.append("length_clamped")
    result.outcome = "publish" if cards else "suppress"
    timings["total_ms"] = (time.perf_counter() - started) * 1000.0
    result.pass_timings_ms = timings
    result.metadata["path"] = "simplified"
    return result


def terminal_result(
    *,
    outcome: str,
    reason_codes: list[str],
    timings: dict[str, float],
    started: float,
    metadata: dict[str, Any] | None = None,
) -> DialecticalResult:
    timings["total_ms"] = (time.perf_counter() - started) * 1000.0
    return DialecticalResult(
        outcome=outcome,  # type: ignore[arg-type]
        reason_codes=list(reason_codes),
        pass_timings_ms=timings,
        metadata=metadata or {},
        quality=QualityReport(passed=False, errors=list(reason_codes)),
    )
