"""Adaptive context budget from existing topic routing signals."""

from __future__ import annotations

CHARS_SAFE_FLOOR = 2000
# Keep adaptive ceiling below ctx_size*0.9 with max_tokens headroom.
HARD_TOPIC_CHARS = 4200
SOFT_TOPIC_CHARS = 3000
LIGHT_TOPIC_CHARS = 2400

_HARD_PRIMARIES = frozenset({"labor_economy", "geopolitics", "social"})
_LIGHT_PRIMARIES = frozenset({"sport", "crime", "disaster", "science"})


def adaptive_max_context_chars(
    *,
    primary: str,
    base_chars: int,
    ctx_size: int,
    max_tokens: int,
) -> int:
    """Scale context budget by topic class without exceeding token headroom."""
    approx_char_cap = max(CHARS_SAFE_FLOOR, int((ctx_size * 0.9 - max_tokens) * 4))
    if primary in _HARD_PRIMARIES:
        target = max(base_chars, HARD_TOPIC_CHARS)
    elif primary in _LIGHT_PRIMARIES:
        target = min(base_chars, LIGHT_TOPIC_CHARS)
    else:
        target = max(base_chars, SOFT_TOPIC_CHARS)
    return int(min(target, approx_char_cap, max(base_chars, CHARS_SAFE_FLOOR)))
