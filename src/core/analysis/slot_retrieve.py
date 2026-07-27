"""Slot retrieval with widen/rerank fallbacks for dialectical orchestration."""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.core.analysis.evidence_brief import EvidenceItem
from src.core.retrieval.arbiter import RetrievalCandidate
from src.core.retrieval.stance_retrieve import dedupe_by_chunk_id

logger = logging.getLogger(__name__)

RetrieveFn = Callable[..., list[RetrievalCandidate]]


def candidate_to_item(
    candidate: RetrievalCandidate,
    *,
    stance_type: str,
    query_used: str,
) -> EvidenceItem:
    return EvidenceItem(
        stance_type=stance_type,
        source_id=candidate.source_id,
        source_path=candidate.source_path,
        chunk_id=candidate.chunk_id,
        text=candidate.text,
        score=float(candidate.score),
        retriever=candidate.retriever,
        query_used=query_used,
        multi_stance=False,
    )


def retrieve_slot_with_fallback(
    *,
    provider: Any,
    query_text: str,
    stance_type: str,
    limit: int,
    widen_factor: int,
    allow_author_fallback: bool,
) -> tuple[list[EvidenceItem], str]:
    """Return (items, fallback_step). Steps: primary|widen|rerank|author|empty."""
    retrieve = getattr(provider, "retrieve_by_stance", None)
    if retrieve is None:
        return [], "empty"

    primary = dedupe_by_chunk_id(
        retrieve(
            query_text,
            stance_types=[stance_type],
            limit=limit,
            apply_internal_multi_query=True,
        )
    )
    if primary:
        return (
            [candidate_to_item(item, stance_type=stance_type, query_used=query_text) for item in primary[:limit]],
            "primary",
        )

    widen_limit = max(limit * widen_factor, limit)
    widened = dedupe_by_chunk_id(
        retrieve(
            query_text,
            stance_types=[stance_type],
            limit=widen_limit,
            apply_internal_multi_query=True,
        )
    )
    if widened:
        return (
            [candidate_to_item(item, stance_type=stance_type, query_used=query_text) for item in widened[:limit]],
            "widen",
        )

    unfiltered_retrieve = getattr(provider, "retrieve", None)
    if callable(unfiltered_retrieve):
        raw = unfiltered_retrieve(query_text)
        filtered = [item for item in raw if item.stance_type == stance_type]
        filtered = sorted(filtered, key=lambda item: item.score, reverse=True)
        if filtered:
            return (
                [
                    candidate_to_item(item, stance_type=stance_type, query_used=query_text)
                    for item in filtered[:limit]
                ],
                "rerank",
            )

    if allow_author_fallback:
        context_fn = getattr(provider, "retrieve_context", None)
        if callable(context_fn):
            result = context_fn(query_text=query_text, author_filter="Ленин")
            text = getattr(result, "context", "") or ""
            if text.strip():
                return (
                    [
                        EvidenceItem(
                            stance_type=stance_type,
                            source_id="legacy_author",
                            source_path="legacy_author_filter",
                            chunk_id="legacy_author_0",
                            text=text[:1200],
                            score=0.0,
                            retriever="legacy",
                            query_used=query_text,
                        )
                    ],
                    "author",
                )
    return [], "empty"
