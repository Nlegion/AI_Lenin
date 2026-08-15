"""Bounded R1 excerpts for DeepSeek quote strict mode (provider-isolated)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.analysis.evidence_brief import EvidenceBrief, EvidenceItem
from src.core.generation.deepseek_excerpt_rank import excerpt_link_score
from src.core.generation.quote_allowlist import QuoteCandidate
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_CONTENT_TOKEN = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

_MAX_EXCERPTS = 6
_MAX_EXCERPT_CHARS = 280
_MIN_LINK_SCORE = 0.12
_SOFT_FALLBACK_TOP_K = 3


@dataclass(frozen=True)
class DeepSeekR1Excerpts:
    candidates: list[QuoteCandidate]
    block: str
    best_link_score: float = 0.0

    @property
    def usable(self) -> bool:
        return bool(self.candidates)


def _content_token_count(text: str) -> int:
    return len(_CONTENT_TOKEN.findall(text))


def _sentences(text: str) -> list[str]:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]
    return parts or [cleaned]


def _clip(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _usable_excerpt(
    text: str,
    *,
    config: QualityPostcheckConfig,
) -> str | None:
    excerpt = _clip(text, max_chars=_MAX_EXCERPT_CHARS)
    if len(excerpt) < int(config.min_quote_chars):
        return None
    if _content_token_count(excerpt) < int(config.min_quote_content_tokens):
        return None
    return excerpt


def _best_excerpt_for_item(
    *,
    item: EvidenceItem,
    news: str,
    concepts_blob: str,
    config: QualityPostcheckConfig,
) -> tuple[float, str, dict[str, float]] | None:
    scored: list[tuple[float, str, dict[str, float]]] = []
    for sentence in _sentences(item.text):
        excerpt = _usable_excerpt(sentence, config=config)
        if not excerpt:
            continue
        metrics = excerpt_link_score(
            news=news,
            excerpt=excerpt,
            query_used=item.query_used or "",
            concepts_blob=concepts_blob,
            retrieval_score=float(item.score or 0.0),
        )
        scored.append((metrics["link_score"], excerpt, metrics))
    if not scored:
        fallback = _usable_excerpt(item.text, config=config)
        if not fallback:
            return None
        metrics = excerpt_link_score(
            news=news,
            excerpt=fallback,
            query_used=item.query_used or "",
            concepts_blob=concepts_blob,
            retrieval_score=float(item.score or 0.0),
        )
        return metrics["link_score"], fallback, metrics
    scored.sort(key=lambda pair: (-pair[0], -len(pair[1])))
    return scored[0]


def build_deepseek_r1_excerpts(
    *,
    brief: EvidenceBrief | None,
    config: QualityPostcheckConfig,
    news_text: str = "",
) -> DeepSeekR1Excerpts:
    """Derive R1-only quote excerpts ranked by news/query/concept link score."""
    items: list[EvidenceItem] = list(brief.r1_core_self) if brief is not None else []
    news = (news_text or "").strip()
    concepts_blob = ""
    if brief is not None:
        concepts_blob = " ".join([*(brief.key_concepts or []), *(brief.axes or [])])
    ranked: list[tuple[float, QuoteCandidate]] = []
    seen: set[str] = set()
    for item in items:
        picked = _best_excerpt_for_item(
            item=item,
            news=news,
            concepts_blob=concepts_blob,
            config=config,
        )
        if not picked:
            continue
        score, excerpt, metrics = picked
        key = excerpt.casefold()
        if key in seen:
            continue
        seen.add(key)
        ranked.append(
            (
                score,
                QuoteCandidate(
                    text=excerpt,
                    chunk_id=item.chunk_id,
                    source_id=item.source_id or None,
                    meta=dict(metrics),
                ),
            )
        )
    ranked.sort(key=lambda pair: (-pair[0], -len(pair[1].text)))
    best_link = ranked[0][0] if ranked else 0.0
    if news and ranked:
        above = [pair for pair in ranked if pair[0] >= _MIN_LINK_SCORE]
        ranked = above if above else ranked[:_SOFT_FALLBACK_TOP_K]
    candidates = [cand for _, cand in ranked[:_MAX_EXCERPTS]]
    if not candidates:
        return DeepSeekR1Excerpts(candidates=[], block="", best_link_score=0.0)
    lines = [
        "Допустимые цитаты из R1 (выбери одну с явной смысловой связью к факту новости; "
        "вставляй дословно):"
    ]
    if best_link < _MIN_LINK_SCORE:
        lines.append(
            "Связь с новостью слабая: если ни одна цитата не объясняет факт — напиши "
            "без кавычек: В предоставленном контексте подходящей цитаты нет."
        )
    for cand in candidates:
        source = cand.source_id or cand.chunk_id
        lines.append(f"- «{cand.text}» ({source})")
    return DeepSeekR1Excerpts(
        candidates=candidates,
        block="\n".join(lines),
        best_link_score=float(best_link),
    )
