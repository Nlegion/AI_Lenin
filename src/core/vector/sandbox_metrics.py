"""Metrics and rank fusion helpers for retrieval sandbox experiments."""

from __future__ import annotations


def reciprocal_rank_fusion(
    rank_lists: dict[str, list[str]],
    retriever_weights: dict[str, float],
    k: int = 60,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for retriever_name, items in rank_lists.items():
        weight = retriever_weights.get(retriever_name, 1.0)
        for rank, item_id in enumerate(items, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    return scores


def apply_source_boost(
    scores: dict[str, float],
    source_stance: dict[str, str],
    boosts: dict[str, float],
) -> dict[str, float]:
    boosted: dict[str, float] = {}
    for source_id, score in scores.items():
        stance = source_stance.get(source_id, "contextual")
        boosted[source_id] = score * boosts.get(stance, 1.0)
    return boosted


def recall_at_k(predictions: list[list[str]], positives: list[str], k: int) -> float:
    if not predictions:
        return 0.0
    hits = 0
    for predicted_ids, positive in zip(predictions, positives):
        if positive in predicted_ids[:k]:
            hits += 1
    return hits / len(predictions)
