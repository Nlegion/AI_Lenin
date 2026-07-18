"""Embedding benchmark metrics and model selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str
    recall_at_5: float
    mean_latency_ms: float
    ram_delta_mb: float
    vram_peak_mb: float | None
    status: str
    notes: str = ""


def _dot(vec_a: list[float], vec_b: list[float]) -> float:
    return sum(left * right for left, right in zip(vec_a, vec_b))


def _norm(vector: list[float]) -> float:
    return sqrt(sum(value * value for value in vector))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    denominator = _norm(vec_a) * _norm(vec_b)
    if denominator == 0:
        return 0.0
    return _dot(vec_a, vec_b) / denominator


def compute_recall_at_k(
    query_embeddings: list[list[float]],
    document_embeddings: list[list[float]],
    positives: list[int],
    k: int,
) -> float:
    if not query_embeddings:
        return 0.0

    hits = 0
    for query_idx, query_vector in enumerate(query_embeddings):
        doc_scores = [
            (doc_idx, cosine_similarity(query_vector, document_embeddings[doc_idx]))
            for doc_idx in range(len(document_embeddings))
        ]
        ranked = sorted(doc_scores, key=lambda item: item[1], reverse=True)[:k]
        ranked_ids = {doc_idx for doc_idx, _ in ranked}
        if positives[query_idx] in ranked_ids:
            hits += 1
    return hits / len(query_embeddings)


def choose_best_model(
    results: list[BenchmarkResult],
    min_recall_at_5: float,
) -> tuple[BenchmarkResult | None, bool]:
    completed = [result for result in results if result.status == "ok"]
    if not completed:
        return None, True

    ranked = sorted(
        completed,
        key=lambda item: (item.recall_at_5, -item.mean_latency_ms),
        reverse=True,
    )
    winner = ranked[0]
    should_fine_tune = winner.recall_at_5 < min_recall_at_5
    return winner, should_fine_tune
