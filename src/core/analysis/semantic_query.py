"""Compose semantic-core retrieval queries with space-aware truncation."""

from __future__ import annotations

from src.core.analysis.semantic_core_config import SemanticCoreConfig
from src.core.analysis.semantic_normalize import normalize_routing


def join_terms_with_budget(
    *,
    terms: list[str],
    max_chars: int,
) -> str:
    """Join terms with single spaces; never mid-term cut; spaces count in budget."""
    parts: list[str] = []
    for term in terms:
        cleaned = term.strip()
        if not cleaned:
            continue
        candidate = cleaned if not parts else f"{' '.join(parts)} {cleaned}"
        if len(candidate) <= max_chars:
            parts.append(cleaned)
            continue
        break
    return " ".join(parts)


def compose_abstract_query(
    *,
    retrieval_terms: list[str],
    news_title: str,
    config: SemanticCoreConfig,
    axes: list[str] | None = None,
) -> str:
    terms = list(retrieval_terms[: config.max_terms_per_topic])
    if config.include_axes_in_semantic_query and axes:
        terms.extend(item for item in axes if item and item not in terms)
    query = join_terms_with_budget(terms=terms, max_chars=config.max_query_chars)
    if not config.include_title_anchor:
        return query
    title_part = normalize_routing(
        news_title,
        normalize_yo_flag=config.normalize_yo_for_routing,
    )[: config.max_title_anchor_chars].strip()
    if not title_part:
        return query
    if not query:
        return join_terms_with_budget(
            terms=[title_part],
            max_chars=config.max_query_chars,
        )
    with_title = f"{query} {title_part}"
    if len(with_title) <= config.max_query_chars:
        return with_title
    # Terms have priority: never drop accepted terms for title.
    return query


def compose_legacy_enriched_query(
    *,
    base_query: str,
    retrieval_terms: list[str],
    config: SemanticCoreConfig,
) -> str:
    terms = list(retrieval_terms[: config.max_terms_per_topic])
    if not terms:
        return base_query
    joined = join_terms_with_budget(terms=terms, max_chars=config.max_query_chars)
    if config.apply_to_legacy is False:
        return base_query
    mode = "prefix"  # documented default for when legacy enrich is enabled
    # Prefer prefix when applying (config field may be added later via YAML).
    if mode == "prefix":
        candidate = f"{joined} {base_query}".strip()
    else:
        candidate = f"{base_query} {joined}".strip()
    if len(candidate) <= config.max_query_chars * 2:
        # Legacy path keeps news anchor; soft char guard uses 2x abstract budget.
        return candidate
    return candidate[: config.max_query_chars * 2].rsplit(" ", 1)[0]
