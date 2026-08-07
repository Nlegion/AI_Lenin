"""Token-safe and phrase-aware pattern matching for NewsGuard."""

from __future__ import annotations

import re

# Patterns that must not match as bare substrings inside longer words.
TOKEN_BOUND_PATTERNS = frozenset({"спорт", "сво"})

# Substring patterns with explicit exclusion contexts (lowercase).
EXCLUDE_CONTEXTS: dict[str, tuple[str, ...]] = {
    "национальн": (
        "национальная компания",
        "национальной компании",
        "национальный проект",
        "национального проекта",
        "национальная экономика",
        "национальной экономики",
    ),
}

SVO_PHRASES = (
    "в рамках сво",
    "ход сво",
    "ходе сво",
    "специальной военной операции",
    "специальная военная операция",
)


def word_boundary_hit(lowered: str, pattern: str) -> bool:
    """True if pattern appears as a whole token (Cyrillic/Latin aware)."""
    escaped = re.escape(pattern.lower())
    return re.search(rf"(?<![а-яёa-z0-9]){escaped}(?![а-яёa-z0-9])", lowered) is not None


def pattern_hits(text: str, patterns: list[str]) -> list[str]:
    """Return matched patterns using token-safe rules for known FP stems."""
    lowered = text.lower()
    hits: list[str] = []
    for pattern in patterns:
        key = pattern.lower().strip()
        if not key:
            continue
        if key in EXCLUDE_CONTEXTS:
            if key in lowered and not any(ctx in lowered for ctx in EXCLUDE_CONTEXTS[key]):
                hits.append(pattern)
            continue
        if key == "сво":
            if svo_token_hit(lowered=lowered):
                hits.append(pattern)
            continue
        if key in TOKEN_BOUND_PATTERNS or key == "спорт":
            if word_boundary_hit(lowered=lowered, pattern=key):
                hits.append(pattern)
            continue
        if key in lowered:
            hits.append(pattern)
    return hits


def svo_token_hit(*, lowered: str) -> bool:
    """Match СВО as a whole token or known military phrases — not «свой/своя»."""
    if any(phrase in lowered for phrase in SVO_PHRASES):
        return True
    return re.search(r"(?<![а-яёa-z0-9])сво(?![а-яёa-z0-9])", lowered) is not None
