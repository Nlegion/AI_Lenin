"""Lenin author payload normalization for Phase 0 / legacy A/B."""

from __future__ import annotations

import re

from src.core.analysis.semantic_normalize import normalize_yo

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")

DEFAULT_LENIN_ALIASES: tuple[str, ...] = (
    "ленин",
    "в.и. ленин",
    "ленин в.и.",
    "владимир ильич ленин",
    "ленин ви",
    "ленин в и",
)

DEFAULT_REJECT_SUBSTRINGS: tuple[str, ...] = (
    "ленинизм",
    "антиленин",
    "анти-ленин",
)


def normalize_author(raw: str | None) -> str:
    if raw is None:
        return ""
    text = str(raw).casefold()
    text = normalize_yo(text)
    text = _NON_WORD.sub(" ", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def is_lenin_author(
    raw: str | None,
    *,
    aliases: list[str] | tuple[str, ...] | None = None,
    reject_substrings: list[str] | tuple[str, ...] | None = None,
) -> bool:
    if raw is None:
        return False
    normalized = normalize_author(raw)
    if not normalized:
        return False
    reject = reject_substrings if reject_substrings is not None else DEFAULT_REJECT_SUBSTRINGS
    for bad in reject:
        if normalize_author(bad) and normalize_author(bad) in normalized:
            return False
    alias_set = {
        normalize_author(item)
        for item in (aliases if aliases is not None else DEFAULT_LENIN_ALIASES)
        if item
    }
    alias_set.discard("")
    if normalized in alias_set:
        return True
    tokens = [token for token in normalized.split() if len(token) > 1]
    return tokens == ["ленин"]
