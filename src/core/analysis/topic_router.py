"""Lexical abstract-topic router for semantic core."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

from src.core.analysis.semantic_core_config import AbstractTopic, SemanticCoreConfig
from src.core.analysis.semantic_normalize import normalize_yo, title_hash

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopicRouteResult:
    dominant_topic_id: str | None
    secondary_topic_ids: list[str]
    retrieval_terms: list[str]
    synthesis_hints: list[str]
    matched_triggers: list[str]
    hint_only: bool
    warnings: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    max_trigger_weight: dict[str, float] = field(default_factory=dict)


def _prepare_text(text: str, *, normalize_yo_flag: bool) -> str:
    lowered = (text or "").casefold()
    if normalize_yo_flag:
        lowered = normalize_yo(lowered)
    return lowered


def route_topics(
    *,
    news_title: str,
    news_content: str,
    config: SemanticCoreConfig,
    run_id: str | None = None,
) -> TopicRouteResult:
    if not config.enabled:
        return TopicRouteResult(
            dominant_topic_id=None,
            secondary_topic_ids=[],
            retrieval_terms=[],
            synthesis_hints=[],
            matched_triggers=[],
            hint_only=False,
            warnings=[],
        )

    haystack = _prepare_text(
        f"{news_title}\n{news_content}",
        normalize_yo_flag=config.normalize_yo_for_routing,
    )
    scores: dict[str, float] = {}
    max_weights: dict[str, float] = {}
    matched_by_topic: dict[str, list[str]] = {}
    topic_by_id: dict[str, AbstractTopic] = {
        topic.topic_id: topic for topic in config.topics
    }

    for topic in config.topics:
        score = 0.0
        max_w = 0.0
        matched: list[str] = []
        for trigger in topic.triggers:
            if trigger.pattern.search(haystack):
                score += trigger.weight
                max_w = max(max_w, trigger.weight)
                matched.append(trigger.text)
        if score > 0:
            scores[topic.topic_id] = score
            max_weights[topic.topic_id] = max_w
            matched_by_topic[topic.topic_id] = matched

    if not scores:
        return TopicRouteResult(
            dominant_topic_id=None,
            secondary_topic_ids=[],
            retrieval_terms=[],
            synthesis_hints=[],
            matched_triggers=[],
            hint_only=False,
            warnings=["semantic_core_no_topic"],
        )

    ranked = sorted(
        scores.keys(),
        key=lambda topic_id: (
            -scores[topic_id],
            -max_weights.get(topic_id, 0.0),
            list(topic_by_id.keys()).index(topic_id),
        ),
    )
    dominant_id = ranked[0]
    secondary = ranked[1 : config.max_topics_logged]
    dominant = topic_by_id[dominant_id]
    warnings: list[str] = []
    if secondary:
        warnings.append("semantic_core_multi_topic")
        logger.info(
            "semantic_core_multi_topic",
            extra={
                "dominant": dominant_id,
                "secondary": secondary,
                "scores": {
                    key: scores[key] for key in ranked[: config.max_topics_logged]
                },
                "max_trigger_weight": {
                    key: max_weights[key] for key in ranked[: config.max_topics_logged]
                },
                "matched_triggers": matched_by_topic.get(dominant_id, []),
                "title_hash": title_hash(
                    news_title,
                    normalize_yo_flag=config.normalize_yo_for_routing,
                ),
                "run_id": run_id,
            },
        )

    if dominant.hint_only or not dominant.retrieval_terms:
        warnings.append("semantic_core_hint_only")
        return TopicRouteResult(
            dominant_topic_id=dominant_id,
            secondary_topic_ids=secondary,
            retrieval_terms=[],
            synthesis_hints=[dominant.synthesis_hint]
            if dominant.synthesis_hint
            else [],
            matched_triggers=matched_by_topic.get(dominant_id, []),
            hint_only=True,
            warnings=warnings,
            scores=scores,
            max_trigger_weight=max_weights,
        )

    terms = list(dominant.retrieval_terms[: config.max_terms_per_topic])
    return TopicRouteResult(
        dominant_topic_id=dominant_id,
        secondary_topic_ids=secondary,
        retrieval_terms=terms,
        synthesis_hints=[dominant.synthesis_hint] if dominant.synthesis_hint else [],
        matched_triggers=matched_by_topic.get(dominant_id, []),
        hint_only=False,
        warnings=warnings,
        scores=scores,
        max_trigger_weight=max_weights,
    )
