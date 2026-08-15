"""Routing text normalization and stable title hashing for semantic core."""

from __future__ import annotations

from hashlib import sha256
import re

_PUNCT_TO_SPACE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")


def normalize_yo(text: str) -> str:
    return text.replace("ё", "е").replace("Ё", "Е")


def normalize_routing(text: str, *, normalize_yo_flag: bool = True) -> str:
    raw = text or ""
    lowered = raw.casefold()
    if normalize_yo_flag:
        lowered = normalize_yo(lowered)
    cleaned = _PUNCT_TO_SPACE.sub(" ", lowered)
    return _MULTI_SPACE.sub(" ", cleaned).strip()


def tokenize_routing(text: str, *, normalize_yo_flag: bool = True) -> list[str]:
    """Whitespace split after normalize_routing only (no spaCy/NLTK)."""
    normalized = normalize_routing(text, normalize_yo_flag=normalize_yo_flag)
    return [token for token in normalized.split() if token]


def title_hash(title: str, *, normalize_yo_flag: bool = True) -> str:
    normalized = normalize_routing(title, normalize_yo_flag=normalize_yo_flag)
    digest = sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:16]


def build_baseline_query(
    *,
    news_title: str,
    news_content: str,
    stopwords: set[str],
    content_token_limit: int,
    short_lead_chars: int = 200,
    normalize_yo_flag: bool = True,
) -> str:
    lead = news_content[:short_lead_chars]
    tokens = tokenize_routing(
        f"{news_title} {lead}",
        normalize_yo_flag=normalize_yo_flag,
    )
    content = [token for token in tokens if token not in stopwords and len(token) > 1]
    return " ".join(content[:content_token_limit]).strip()
