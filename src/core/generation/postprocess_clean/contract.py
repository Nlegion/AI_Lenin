"""Typed contract for the unified answer postprocess module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

PostProcessPhase = Literal["pre_guard", "post_guard"]
PostProcessStatus = Literal["clean", "blocked", "needs_review"]
PostProcessCleanMode = Literal["off", "shadow", "live"]

_TRIAD_PATTERNS = (
    r"(?:^|\s)\*{0,2}факт\*{0,2}\s*:",
    r"(?:^|\s)\*{0,2}механизм\*{0,2}\s*:",
    r"(?:^|\s)\*{0,2}вывод\*{0,2}\s*:",
)


@dataclass
class PostProcessInput:
    raw_text: str
    phase: PostProcessPhase
    combat_sensitive: bool = False
    item_id: str | None = None
    skip_structure_enforce: bool = False
    metadata: dict[str, Any] | None = None
    config: QualityPostcheckConfig | None = None


@dataclass
class PostProcessResult:
    cleaned_text: str
    status: PostProcessStatus
    codes: list[str] = field(default_factory=list)
    error_details: Optional[str] = None
    postprocess_hard_fail: bool = False
    structure_error: bool = False
    integrity_error: bool = False
    integrity_codes: list[str] = field(default_factory=list)
    body_cleanup_codes: list[str] = field(default_factory=list)
    integrity_enforce_mode: str = "soft"

    def to_legacy_metadata(self) -> dict[str, Any]:
        """Keys consumed by quality_hooks / publishability / monitoring."""
        return {
            "body_cleanup_codes": list(self.body_cleanup_codes),
            "integrity_codes": list(self.integrity_codes),
            "integrity_error": self.integrity_error,
            "postprocess_hard_fail": self.postprocess_hard_fail,
            "integrity_enforce_mode": self.integrity_enforce_mode,
            "postprocess_status": self.status,
            "postprocess_codes": list(self.codes),
            "structure_error": self.structure_error,
        }


def has_required_triad(text: str) -> bool:
    import re

    return all(
        re.search(pattern=pattern, string=text, flags=re.IGNORECASE) is not None
        for pattern in _TRIAD_PATTERNS
    )


def map_postprocess_status(
    *,
    postprocess_hard_fail: bool,
    structure_error: bool,
    deny: bool = False,
) -> PostProcessStatus:
    """Adapter over existing flags. Does not replace NewsGuard blocked."""
    if postprocess_hard_fail or deny:
        return "blocked"
    if structure_error:
        return "needs_review"
    return "clean"


def resolve_clean_mode(config: QualityPostcheckConfig | None) -> PostProcessCleanMode:
    raw = str(getattr(config, "postprocess_clean_mode", "live") or "live").lower()
    if raw in {"off", "shadow", "live"}:
        return raw  # type: ignore[return-value]
    return "live"
