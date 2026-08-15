"""News↔excerpt link scoring for DeepSeek R1 quote selection."""

from __future__ import annotations

from src.core.generation.quote_mode import content_lemmas

# News surface cues → Lenin-lexicon bridges (for low lexical overlap).
_THEME_BRIDGES: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (
        frozenset({"война", "конфликт", "удар", "военн", "обстрел", "фронт"}),
        frozenset({"империализм", "война", "вооруж", "держав", "захват"}),
    ),
    (
        frozenset({"цена", "инфляц", "газ", "энерг", "тариф", "экономик", "рынок"}),
        frozenset({"капитал", "монопол", "кризис", "издержк", "хозяйств", "труд"}),
    ),
    (
        frozenset({"рейтинг", "канцлер", "парламент", "правитель", "выбор", "опрос"}),
        frozenset(
            {"буржуаз", "масс", "пролетариат", "оппортунизм", "реформизм", "парламент"}
        ),
    ),
    (
        frozenset({"диплом", "посол", "визит", "форум", "союз", "интеграц", "санкц"}),
        frozenset({"империализм", "держав", "дипломат", "договор", "буржуаз"}),
    ),
    (
        frozenset(
            {"наук", "учен", "филолог", "культур", "музей", "музык", "фестиваль"}
        ),
        frozenset({"идеолог", "просвещ", "класс", "культур", "агитац", "пропаганд"}),
    ),
)


def _prefix_hits(haystack: str, prefixes: frozenset[str]) -> bool:
    lowered = (haystack or "").casefold()
    return any(prefix in lowered for prefix in prefixes)


def theme_bridge_bonus(*, news: str, excerpt: str) -> float:
    for news_cues, excerpt_cues in _THEME_BRIDGES:
        if _prefix_hits(news, news_cues) and _prefix_hits(excerpt, excerpt_cues):
            return 0.18
    return 0.0


def lemma_coverage(*, left: str, right: str) -> float:
    """Share of right lemmas also present in left."""
    right_toks = content_lemmas(right)
    if not right_toks:
        return 0.0
    left_toks = content_lemmas(left)
    if not left_toks:
        return 0.0
    return len(left_toks & right_toks) / len(right_toks)


def excerpt_link_score(
    *,
    news: str,
    excerpt: str,
    query_used: str = "",
    concepts_blob: str = "",
    retrieval_score: float = 0.0,
) -> dict[str, float]:
    news_cov = lemma_coverage(left=news, right=excerpt) if news else 0.0
    query_cov = (
        lemma_coverage(left=query_used, right=excerpt) if query_used.strip() else 0.0
    )
    concept_cov = (
        lemma_coverage(left=concepts_blob, right=excerpt)
        if concepts_blob.strip()
        else 0.0
    )
    theme = theme_bridge_bonus(news=news, excerpt=excerpt)
    # Retrieval scores are not guaranteed in [0, 1]; clamp contribution.
    retrieval = min(1.0, max(0.0, float(retrieval_score)))
    total = news_cov + 0.45 * query_cov + 0.35 * concept_cov + theme + 0.05 * retrieval
    return {
        "news_coverage": round(news_cov, 4),
        "query_coverage": round(query_cov, 4),
        "concept_coverage": round(concept_cov, 4),
        "theme_bonus": round(theme, 4),
        "link_score": round(total, 4),
    }
