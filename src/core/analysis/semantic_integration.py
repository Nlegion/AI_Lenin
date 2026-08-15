"""Apply semantic-core routing to dialectical / legacy query composition."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

from src.core.analysis.semantic_core_config import (
    SemanticCoreConfig,
    load_semantic_core_config,
)
from src.core.analysis.semantic_query import compose_abstract_query
from src.core.analysis.topic_router import TopicRouteResult, route_topics

logger = logging.getLogger(__name__)

# Follow-up: cache first abstract retrieve embedding/results to avoid full double-fetch on empty R1.


def load_semantic_config(path: Path | None = None) -> SemanticCoreConfig:
    return load_semantic_core_config(path=path)


def maybe_route(
    *,
    news_title: str,
    news_content: str,
    config: SemanticCoreConfig,
    run_id: str | None,
) -> TopicRouteResult | None:
    if not config.enabled:
        return None
    return route_topics(
        news_title=news_title,
        news_content=news_content,
        config=config,
        run_id=run_id,
    )


def dialectical_uses_abstract(
    *,
    semantic: SemanticCoreConfig,
    dialectical_enabled: bool,
    route: TopicRouteResult | None,
) -> bool:
    return bool(
        semantic.enabled
        and dialectical_enabled
        and semantic.apply_to_dialectical
        and route is not None
        and not route.hint_only
        and route.retrieval_terms
    )


def apply_abstract_slot_queries(
    *,
    route: TopicRouteResult,
    semantic: SemanticCoreConfig,
    news_title: str,
    axes: list[str],
    modality: dict[str, str],
    include_modality_suffix: bool,
) -> dict[str, str]:
    base = compose_abstract_query(
        retrieval_terms=route.retrieval_terms,
        news_title=news_title,
        config=semantic,
        axes=axes if semantic.include_axes_in_semantic_query else None,
    )
    queries: dict[str, str] = {}
    for slot, suffix in modality.items():
        if include_modality_suffix and suffix.strip():
            queries[slot] = f"{base} {suffix.strip()}".strip()
        else:
            queries[slot] = base
    return queries


def trace_from_route(route: TopicRouteResult) -> dict[str, Any]:
    return {
        "semantic_core_dominant": route.dominant_topic_id,
        "semantic_core_secondary": list(route.secondary_topic_ids),
        "semantic_core_terms": list(route.retrieval_terms),
        "semantic_core_hint_only": route.hint_only,
        "matched_triggers": list(route.matched_triggers),
        "semantic_core_multi_topic": "semantic_core_multi_topic" in route.warnings,
    }


def mark_fallback(
    *,
    trace: dict[str, Any],
    elapsed_ms: float,
    exhausted: bool = False,
) -> None:
    trace["semantic_fallback"] = True
    trace["semantic_fallback_elapsed_ms"] = round(elapsed_ms, 2)
    if exhausted:
        trace["semantic_fallback_exhausted"] = True
        logger.warning(
            "semantic_core_empty_r1_fallback_exhausted elapsed_ms=%.2f",
            elapsed_ms,
        )
    else:
        logger.warning(
            "semantic_core_empty_r1_fallback elapsed_ms=%.2f",
            elapsed_ms,
        )


def timed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def legacy_enable_decision(
    *,
    author_known_rate: float,
    author_known_rate_min: float,
    human_scores_available: bool,
) -> bool:
    """Whether apply_to_legacy may be considered for enablement."""
    if author_known_rate < author_known_rate_min and not human_scores_available:
        return False
    return True


def cliche_gate_blocks_enable(
    *,
    warn_rate_off: float,
    warn_rate_on: float,
    max_ratio: float,
    min_delta_pp: float,
) -> bool:
    ratio_hit = warn_rate_on > warn_rate_off * max_ratio
    delta_hit = (warn_rate_on - warn_rate_off) > (min_delta_pp / 100.0)
    return bool(ratio_hit and delta_hit)
