"""Warn-only news groundedness heuristic (keyterm overlap)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9\-]{4,}")


@dataclass
class GroundednessResult:
    grounded: bool
    matched_keyterms: list[str]
    overlap_rate: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "matched_keyterms": list(self.matched_keyterms)[:8],
            "overlap_rate": round(self.overlap_rate, 4),
            "ungrounded_news_warn": (not self.grounded),
        }


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN.finditer(text or "")}


def news_groundedness(*, analysis: str, news_title: str, news_content: str) -> GroundednessResult:
    news_tokens = _tokens(f"{news_title} {news_content}")
    answer_tokens = _tokens(analysis)
    if not news_tokens or not answer_tokens:
        return GroundednessResult(grounded=False, matched_keyterms=[], overlap_rate=0.0)
    matched = sorted(news_tokens & answer_tokens)
    overlap = len(matched) / max(len(news_tokens), 1)
    grounded = bool(matched) or overlap >= 0.08
    return GroundednessResult(
        grounded=grounded,
        matched_keyterms=matched[:12],
        overlap_rate=overlap,
    )
