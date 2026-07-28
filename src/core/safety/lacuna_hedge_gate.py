"""Warn-only lacuna-hedge gate: non-mutating, fail-open."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re

from src.core.settings.gate_constants import LACUNA_HEDGE_CODE

logger = logging.getLogger(__name__)

# Shared with prompt bans / eval (substring / phrase patterns, case-insensitive).
LACUNA_HEDGE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ленин\s+не\s+писал",
        r"ленин\s+не\s+упоминал",
        r"ленин\s+не\s+касался",
        r"в\s+текстах\s+ленина\s+нет\s+прямого\s+ответа",
        r"классик\s+не\s+обращался",
        r"классик\s+не\s+касался",
        r"в\s+корпусе\s+нет\s+цитат",
        r"не\s+оставил\s+(прямых\s+)?высказываний",
        r"в\s+наследии\s+отсутствует",
        r"нет\s+дословных\s+высказываний",
    )
)


@dataclass(frozen=True)
class LacunaHedgeGateResult:
    blocked: bool
    warn_only: bool
    reason_codes: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    skipped: bool = False
    mode: str = "warn_only"

    def to_metadata(self) -> dict:
        return {
            "mode": self.mode,
            "blocked": self.blocked,
            "skipped": self.skipped,
            "lacuna_hedge_warn": bool(self.reason_codes),
            "reason_codes": list(self.reason_codes),
            "matched_patterns": list(self.matched_patterns),
        }


def lacuna_hedge_gate(*, analysis: str) -> LacunaHedgeGateResult:
    try:
        text = analysis or ""
        matched: list[str] = []
        for pattern in LACUNA_HEDGE_PATTERNS:
            found = pattern.search(text)
            if found:
                matched.append(found.group(0))
        if not matched:
            return LacunaHedgeGateResult(blocked=False, warn_only=True)
        return LacunaHedgeGateResult(
            blocked=False,
            warn_only=True,
            reason_codes=[LACUNA_HEDGE_CODE],
            matched_patterns=matched,
        )
    except Exception:  # noqa: BLE001
        logger.exception("lacuna_hedge_gate_failed")
        return LacunaHedgeGateResult(blocked=False, warn_only=True, skipped=True)
