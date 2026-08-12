"""Extractive PrincipleCards from EvidenceBrief (no LLM invention)."""

from __future__ import annotations

import hashlib
import re

from src.core.analysis.evidence_brief import EvidenceBrief, EvidenceItem
from src.core.dialectics.config import DialecticalReasoningConfig
from src.core.dialectics.schemas import PrincipleCard

_WS = re.compile(r"\s+")


def _stable_id(chunk_id: str, quote: str) -> str:
    digest = hashlib.sha1(f"{chunk_id}|{quote}".encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"pc-{digest}"


def _quote_from_item(item: EvidenceItem, *, max_quote_chars: int) -> str:
    text = _WS.sub(" ", (item.text or "").strip())
    if len(text) <= max_quote_chars:
        return text
    return text[:max_quote_chars].rstrip()


def _title_from_quote(quote: str) -> str:
    snippet = quote[:80].strip()
    return snippet if snippet else "principle"


def _cards_from_items(
    items: list[EvidenceItem],
    *,
    limit: int,
    max_quote_chars: int,
) -> list[PrincipleCard]:
    cards: list[PrincipleCard] = []
    for item in items[:limit]:
        quote = _quote_from_item(item, max_quote_chars=max_quote_chars)
        if not quote:
            continue
        # Enforce extractive invariant: quote must be substring of original text.
        if quote not in item.text and quote not in _WS.sub(" ", item.text):
            # Normalized whitespace mismatch — still accept if all tokens present
            if quote.casefold() not in item.text.casefold():
                continue
        cards.append(
            PrincipleCard(
                principle_id=_stable_id(item.chunk_id, quote),
                title=_title_from_quote(quote),
                quote=quote,
                chunk_id=item.chunk_id,
                stance_type=item.stance_type,
                source_path=item.source_path or item.source_id,
                inferred=False,
                score=float(item.score),
            )
        )
    return cards


def build_principle_cards(
    brief: EvidenceBrief,
    *,
    config: DialecticalReasoningConfig,
) -> list[PrincipleCard]:
    per_slot = config.max_principles_per_slot
    cards = [
        *_cards_from_items(
            brief.r1_core_self,
            limit=per_slot,
            max_quote_chars=config.max_quote_chars,
        ),
        *_cards_from_items(
            brief.r2_influence_agree,
            limit=per_slot,
            max_quote_chars=config.max_quote_chars,
        ),
        *_cards_from_items(
            brief.r3_influence_critical,
            limit=per_slot,
            max_quote_chars=config.max_quote_chars,
        ),
    ]
    return cards[: config.max_principles]


def cards_by_stance(cards: list[PrincipleCard]) -> dict[str, list[PrincipleCard]]:
    grouped: dict[str, list[PrincipleCard]] = {
        "core_self": [],
        "influence_agree": [],
        "influence_critical": [],
    }
    for card in cards:
        grouped.setdefault(card.stance_type, []).append(card)
    return grouped
