"""Anti-cliché gate: non-mutating, fail-open, skip-inside when brief is absent."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

from src.core.analysis.jaccard_metrics import jaccard_overlap, tokenize_for_jaccard
from src.core.safety.anti_cliche_config import AntiClicheConfig, load_anti_cliche_config
from src.core.settings.gate_constants import (
    CLICHE_CODE_LEXICON_DENSE,
    CLICHE_CODE_LOW_R1_OVERLAP,
    CLICHE_CODE_NO_R1,
    CLICHE_CODE_SKIPPED_NO_BRIEF,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClicheGateResult:
    blocked: bool
    warn_only: bool
    reason_codes: list[str] = field(default_factory=list)
    r1_jaccard: float = 0.0
    lexicon_hits: int = 0
    skipped: bool = False
    skip_reason: str | None = None
    mode: str = "warn_only"

    def to_metadata(self) -> dict:
        return {
            "mode": self.mode,
            "blocked": self.blocked,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "r1_jaccard": self.r1_jaccard,
            "lexicon_hits": self.lexicon_hits,
            "reason_codes": list(self.reason_codes),
        }


def _has_quote_anchor(*, analysis: str, phrases: tuple[str, ...]) -> bool:
    lowered = analysis.casefold()
    return any(phrase in lowered for phrase in phrases)


def _evaluate_cliche_gate(
    *,
    analysis: str,
    brief_present: bool,
    r1_text: str,
    r1_count: int,
    config: AntiClicheConfig,
) -> ClicheGateResult:
    warn_only = config.warn_only
    mode = config.mode
    if not brief_present:
        return ClicheGateResult(
            blocked=False,
            warn_only=warn_only,
            reason_codes=[CLICHE_CODE_SKIPPED_NO_BRIEF],
            skipped=True,
            skip_reason="brief_is_none",
            mode=mode,
        )

    tokens = tokenize_for_jaccard(text=analysis)
    cliche_hits = tokens & config.lexicon
    lexicon_hits = len(cliche_hits)
    dense = lexicon_hits >= config.lexicon_density_min_hits
    overlap = (
        jaccard_overlap(left_text=analysis, right_text=r1_text) if r1_text else 0.0
    )
    reasons: list[str] = []

    if r1_count == 0:
        # quote_anchor never clears cliche_no_r1
        if dense:
            reasons.append(CLICHE_CODE_NO_R1)
        return ClicheGateResult(
            blocked=bool(reasons) and not warn_only,
            warn_only=warn_only,
            reason_codes=reasons,
            r1_jaccard=overlap,
            lexicon_hits=lexicon_hits,
            mode=mode,
        )

    anchored = _has_quote_anchor(analysis=analysis, phrases=config.quote_anchor_phrases)
    if anchored:
        return ClicheGateResult(
            blocked=False,
            warn_only=warn_only,
            reason_codes=[],
            r1_jaccard=overlap,
            lexicon_hits=lexicon_hits,
            mode=mode,
        )

    low_overlap = overlap < config.min_r1_jaccard
    if low_overlap and dense:
        reasons.append(CLICHE_CODE_LOW_R1_OVERLAP)
        reasons.append(CLICHE_CODE_LEXICON_DENSE)

    return ClicheGateResult(
        blocked=bool(reasons) and not warn_only,
        warn_only=warn_only,
        reason_codes=reasons,
        r1_jaccard=overlap,
        lexicon_hits=lexicon_hits,
        mode=mode,
    )


def cliche_gate(
    *,
    analysis: str,
    brief_present: bool,
    r1_text: str = "",
    r1_count: int = 0,
    config: AntiClicheConfig | None = None,
    config_path: str | None = None,
) -> ClicheGateResult:
    """Evaluate cliché signals. Does not modify analysis. Fail-open on errors."""
    try:
        resolved = config or load_anti_cliche_config(path=config_path)
        return _evaluate_cliche_gate(
            analysis=analysis,
            brief_present=brief_present,
            r1_text=r1_text,
            r1_count=r1_count,
            config=resolved,
        )
    except Exception as exc:  # noqa: BLE001 — fail-open contract
        logger.exception("cliche_gate_failed")
        return ClicheGateResult(
            blocked=False,
            warn_only=True,
            reason_codes=[],
            skipped=True,
            skip_reason=f"error: {type(exc).__name__}",
            mode="warn_only",
        )
