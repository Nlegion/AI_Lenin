"""Stance-filtered retrieval helpers for dialectical slots."""

from __future__ import annotations

from qdrant_client import models

from src.core.retrieval.arbiter import RetrievalCandidate, weighted_rrf


def stance_filter(stance_types: list[str]) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="stance_type",
                match=models.MatchAny(any=stance_types),
            )
        ]
    )


def merge_slot_candidates(
    *,
    candidates: list[RetrievalCandidate],
    retriever_weights: dict[str, float],
    rrf_k: int,
    limit: int,
) -> list[RetrievalCandidate]:
    """RRF merge without stance boost (stance already hard-filtered)."""
    if not candidates:
        return []
    scores = weighted_rrf(
        candidates=candidates,
        retriever_weights=retriever_weights,
        rrf_k=rrf_k,
    )
    best_by_chunk: dict[str, RetrievalCandidate] = {}
    for candidate in candidates:
        current = best_by_chunk.get(candidate.chunk_id)
        if current is None or candidate.score > current.score:
            best_by_chunk[candidate.chunk_id] = candidate
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    final: list[RetrievalCandidate] = []
    for chunk_id, _ in ordered:
        item = best_by_chunk.get(chunk_id)
        if item is None:
            continue
        final.append(item)
        if len(final) >= limit:
            break
    return final


def dedupe_by_chunk_id(items: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    seen: set[str] = set()
    result: list[RetrievalCandidate] = []
    for item in items:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        result.append(item)
    return result
