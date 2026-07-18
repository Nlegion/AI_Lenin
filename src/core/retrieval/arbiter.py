"""Arbiter for multi-source retrieval fusion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCandidate:
    source_id: str
    chunk_id: str
    text: str
    stance_type: str
    source_path: str
    retriever: str
    rank: int
    score: float


def weighted_rrf(candidates: list[RetrievalCandidate], retriever_weights: dict[str, float], rrf_k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for candidate in candidates:
        weight = retriever_weights.get(candidate.retriever, 1.0)
        scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + weight / (rrf_k + candidate.rank)
    return scores


def apply_stance_boost(scores: dict[str, float], chunk_to_stance: dict[str, str], source_boosts: dict[str, float]) -> dict[str, float]:
    boosted: dict[str, float] = {}
    for chunk_id, score in scores.items():
        stance = chunk_to_stance.get(chunk_id, "contextual")
        boosted[chunk_id] = score * source_boosts.get(stance, 1.0)
    return boosted


def enforce_core_self_presence(
    ordered_chunk_ids: list[str],
    chunk_to_stance: dict[str, str],
) -> list[str]:
    top_ten = ordered_chunk_ids[:10]
    if any(chunk_to_stance.get(chunk_id) == "core_self" for chunk_id in top_ten):
        return ordered_chunk_ids
    for chunk_id in ordered_chunk_ids:
        if chunk_to_stance.get(chunk_id) == "core_self":
            reordered = [chunk_id] + [item for item in ordered_chunk_ids if item != chunk_id]
            return reordered
    return ordered_chunk_ids


def rerank_with_alpha(
    merged_scores: dict[str, float],
    reranker_scores: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    final_scores: dict[str, float] = {}
    for chunk_id, merged in merged_scores.items():
        rerank = reranker_scores.get(chunk_id, 0.0)
        final_scores[chunk_id] = (1 - alpha) * merged + alpha * rerank
    return final_scores
