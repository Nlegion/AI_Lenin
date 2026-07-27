"""Jaccard overlap metrics for dialectical dry-run evaluation."""

from __future__ import annotations

import re

from src.core.settings.dialectical_constants import JACCARD_STOPWORDS


_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}")


def tokenize_for_jaccard(text: str) -> set[str]:
    """Tokenize using ONLY JACCARD_STOPWORDS — never domain denylist."""
    tokens = {match.group(0).casefold() for match in _TOKEN_RE.finditer(text)}
    return {token for token in tokens if token not in JACCARD_STOPWORDS}


def jaccard_overlap(left_text: str, right_text: str) -> float:
    left = tokenize_for_jaccard(text=left_text)
    right = tokenize_for_jaccard(text=right_text)
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
