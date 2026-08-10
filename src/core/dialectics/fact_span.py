"""Deterministic fact span from news (no LLM)."""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def extract_fact_span(*, news_title: str, news_content: str, max_chars: int = 320) -> str:
    title = _WS.sub(" ", (news_title or "").strip())
    body = _WS.sub(" ", (news_content or "").strip())
    if not body:
        return title[:max_chars]
    lead = body[: max(0, max_chars - len(title) - 2)].rstrip()
    if title and lead:
        return f"{title}. {lead}".strip()[:max_chars]
    return (title or lead)[:max_chars]


def entity_tokens_in_news(fact: str, news_blob: str) -> bool:
    """Loose check: at least one meaningful token from fact appears in news."""
    tokens = [t for t in re.findall(r"[A-Za-zА-Яа-яЁё0-9\-]{4,}", fact.casefold())]
    if not tokens:
        return True
    blob = news_blob.casefold()
    return any(token in blob for token in tokens[:8])
