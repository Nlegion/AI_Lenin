"""Lexical overlap and quote-span helpers for conditional quote mode."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_STOP = frozenset(
    "и в во на с со по о об от до из за для не ни что это как его ее их они мы вы ты я".split()
)

QUOTE_CHARS = re.compile(r"[«»\"„“”]")


@dataclass(frozen=True)
class ChunkOverlap:
    chunk_id: str
    score: float
    text: str
    overlap: float
    has_quote: bool


def content_lemmas(text: str) -> set[str]:
    """Lightweight content tokens (optional pymorphy; fallback stem=token)."""
    tokens = re.findall(r"[а-яёa-z0-9]+", text.lower())
    kept = {t for t in tokens if (t.isdigit() or len(t) >= 3) and t not in _STOP}
    try:
        import pymorphy3  # type: ignore

        morph = pymorphy3.MorphAnalyzer()
        out: set[str] = set()
        for tok in kept:
            if tok.isdigit():
                out.add(tok)
                continue
            parsed = morph.parse(tok)[0]
            if parsed.tag.POS in {"NOUN", "VERB", "INFN", "ADJF", "ADJS", "NUMR", "NPRO"} or tok[0].isupper():
                out.add(parsed.normal_form)
            elif len(tok) >= 4:
                out.add(parsed.normal_form)
        return out or kept
    except (ImportError, ValueError, AttributeError):
        return kept


def lexical_overlap(news: str, chunk: str) -> float:
    news_toks = content_lemmas(news)
    if not news_toks:
        return 0.0
    chunk_toks = content_lemmas(chunk)
    return len(news_toks & chunk_toks) / max(1, len(news_toks))


def has_quote_span(text: str) -> bool:
    if QUOTE_CHARS.search(text):
        return True
    return bool(re.search(r"том\s*\d+|стр\.?\s*\d+", text, flags=re.IGNORECASE))


def answer_has_quotes(text: str) -> bool:
    return QUOTE_CHARS.search(text) is not None


def strip_quotes(text: str) -> str:
    return QUOTE_CHARS.sub("", text)


def select_quote_mode(
    *,
    news: str,
    chunks: list[tuple[str, float, str]],
    top_k: int = 3,
    overlap_threshold: float = 0.15,
) -> tuple[str, list[ChunkOverlap]]:
    """Return mode 'quote'|'principles' and scored chunk overlaps.

    chunks: list of (chunk_id, score, text) sorted by score descending preferred.
    """
    ranked = sorted(chunks, key=lambda item: item[1], reverse=True)[:top_k]
    overlaps: list[ChunkOverlap] = []
    for chunk_id, score, text in ranked:
        ov = lexical_overlap(news=news, chunk=text)
        overlaps.append(
            ChunkOverlap(
                chunk_id=chunk_id,
                score=score,
                text=text,
                overlap=ov,
                has_quote=has_quote_span(text),
            )
        )
    for item in overlaps:
        if item.has_quote and item.overlap >= overlap_threshold:
            return "quote", overlaps
    return "principles", overlaps


def chunk_trace_payload(chunks: list[tuple[str, float, str]], *, text_cap: int = 240) -> list[dict]:
    out: list[dict] = []
    for chunk_id, score, text in chunks[:5]:
        clipped = text[:text_cap]
        out.append(
            {
                "chunk_id": chunk_id,
                "score": score,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                "text": clipped,
            }
        )
    return out
