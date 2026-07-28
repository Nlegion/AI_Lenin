"""Anti-anachronism gate: first-person × modern-tech with quote/attribution exemption."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import logging
from pathlib import Path
import re

import yaml

from src.core.settings.gate_constants import ANACHRONISM_CODE_FIRST_PERSON_TECH

logger = logging.getLogger(__name__)

_QUOTE_RE = re.compile(r"[«\"]([^»\"]+)[»\"]")


@dataclass(frozen=True)
class AnachronismConfig:
    mode: str = "warn_only"
    modern_tech_terms: tuple[str, ...] = ()
    first_person_cues: tuple[str, ...] = ()
    attribution_cues: tuple[str, ...] = ()

    @property
    def warn_only(self) -> bool:
        return self.mode != "block"


@dataclass(frozen=True)
class AnachronismGateResult:
    blocked: bool
    warn_only: bool
    reason_codes: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None
    mode: str = "warn_only"

    def to_metadata(self) -> dict:
        return {
            "mode": self.mode,
            "blocked": self.blocked,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "reason_codes": list(self.reason_codes),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=4)
def load_anachronism_config(path: str | None = None) -> AnachronismConfig:
    config_path = Path(path) if path else _repo_root() / "config" / "anachronism.yaml"
    if not config_path.is_file():
        return AnachronismConfig()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AnachronismConfig(
        mode=str(payload.get("mode", "warn_only")),
        modern_tech_terms=tuple(str(x).casefold() for x in payload.get("modern_tech_terms", [])),
        first_person_cues=tuple(str(x).casefold() for x in payload.get("first_person_cues", [])),
        attribution_cues=tuple(str(x).casefold() for x in payload.get("attribution_cues", [])),
    )


def _strip_quoted_spans(text: str) -> str:
    return _QUOTE_RE.sub(" ", text)


def _near_attribution(*, text: str, index: int, cues: tuple[str, ...], window: int = 40) -> bool:
    start = max(0, index - window)
    prefix = text[start:index]
    return any(cue in prefix for cue in cues)


def _evaluate(
    *,
    analysis: str,
    config: AnachronismConfig,
) -> AnachronismGateResult:
    warn_only = config.warn_only
    working = _strip_quoted_spans(analysis).casefold()
    reasons: list[str] = []
    for cue in config.first_person_cues:
        cue_idx = working.find(cue)
        if cue_idx < 0:
            continue
        for term in config.modern_tech_terms:
            term_idx = working.find(term)
            if term_idx < 0:
                continue
            # Require cue and term in the same rough neighborhood
            if abs(term_idx - cue_idx) > 80:
                continue
            if _near_attribution(text=working, index=min(cue_idx, term_idx), cues=config.attribution_cues):
                continue
            reasons.append(ANACHRONISM_CODE_FIRST_PERSON_TECH)
            break
        if reasons:
            break

    return AnachronismGateResult(
        blocked=bool(reasons) and not warn_only,
        warn_only=warn_only,
        reason_codes=reasons,
        mode=config.mode,
    )


def anachronism_gate(
    *,
    analysis: str,
    config: AnachronismConfig | None = None,
    config_path: str | None = None,
) -> AnachronismGateResult:
    """Non-mutating; always callable; fail-open on errors."""
    try:
        resolved = config or load_anachronism_config(path=config_path)
        return _evaluate(analysis=analysis, config=resolved)
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.exception("anachronism_gate_failed")
        return AnachronismGateResult(
            blocked=False,
            warn_only=True,
            skipped=True,
            skip_reason=f"error: {type(exc).__name__}",
            mode="warn_only",
        )
