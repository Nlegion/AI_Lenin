"""Simple anti-cliché gate for dialectical synthesis (Phase 4 / H1)."""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.core.analysis.jaccard_metrics import jaccard_overlap, tokenize_for_jaccard

CLICHE_LEXICON = frozenset(
    {
        "революция",
        "эксплуатация",
        "пролетариат",
        "буржуазия",
        "классовая",
        "империализм",
        "диктатура",
    }
)


@dataclass(frozen=True)
class ClicheGateResult:
    blocked: bool
    warn_only: bool
    reason_codes: list[str]
    r1_jaccard: float


def cliche_gate(
    *,
    analysis: str,
    r1_text: str,
    r1_count: int,
    warn_only: bool = True,
    min_r1_jaccard: float = 0.02,
) -> ClicheGateResult:
    """Warn/block when answer looks cliché-only and lacks R1 provenance."""
    reasons: list[str] = []
    tokens = tokenize_for_jaccard(text=analysis)
    cliche_hits = tokens & CLICHE_LEXICON
    overlap = jaccard_overlap(left_text=analysis, right_text=r1_text) if r1_text else 0.0
    if r1_count == 0:
        reasons.append("r1_count_zero")
    if overlap < min_r1_jaccard:
        reasons.append("low_r1_overlap")
    if len(cliche_hits) >= 3 and overlap < min_r1_jaccard:
        reasons.append("cliche_lexicon_dense")
    blocked = bool(reasons) and not warn_only
    return ClicheGateResult(
        blocked=blocked,
        warn_only=warn_only,
        reason_codes=reasons,
        r1_jaccard=overlap,
    )


def strip_non_letters(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
