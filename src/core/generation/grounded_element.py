"""At-least-one grounded element check (quote or R1 keyword overlap)."""

from __future__ import annotations

from src.core.generation.quote_mode import content_lemmas


def has_r1_keyword_overlap(*, analysis: str, r1_text: str, min_hits: int = 1) -> bool:
    if not analysis.strip() or not r1_text.strip():
        return False
    a = content_lemmas(analysis)
    b = content_lemmas(r1_text)
    if not a or not b:
        return False
    return len(a & b) >= min_hits
