"""Normalization helpers for quote grounding (deterministic, no fuzzy paraphrase)."""

from __future__ import annotations

import re
import unicodedata

_QUOTE_UNIFY = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "„": '"',
        "“": '"',
        "”": '"',
        "‟": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        "…": "...",
        "ё": "е",
        "Ё": "е",
    }
)
_WS = re.compile(r"\s+")


def normalize_for_grounding(text: str) -> str:
    """NFKC, unify quotes/dashes, ё→е, casefold, collapse whitespace."""
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", text)
    value = value.translate(_QUOTE_UNIFY)
    value = value.casefold()
    value = _WS.sub(" ", value).strip()
    return value
