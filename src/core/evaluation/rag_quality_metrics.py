"""Quality metrics for retrieval and ideology consistency evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import statistics


IDEOLOGY_TERMS = {
    "класс",
    "пролетариат",
    "буржуазия",
    "империализм",
    "материализм",
    "диалектика",
    "эксплуатация",
    "капитал",
}


def recall_at_k(predictions: list[list[str]], positives: list[str], k: int) -> float:
    if not predictions:
        return 0.0
    hits = 0
    for predicted, positive in zip(predictions, positives):
        if positive in predicted[:k]:
            hits += 1
    return hits / len(predictions)


def mrr_at_k(predictions: list[list[str]], positives: list[str], k: int) -> float:
    if not predictions:
        return 0.0
    total = 0.0
    for predicted, positive in zip(predictions, positives):
        reciprocal = 0.0
        for idx, source_id in enumerate(predicted[:k], start=1):
            if source_id == positive:
                reciprocal = 1.0 / idx
                break
        total += reciprocal
    return total / len(predictions)


def ndcg_at_k(predictions: list[list[str]], positives: list[str], k: int) -> float:
    if not predictions:
        return 0.0
    total = 0.0
    for predicted, positive in zip(predictions, positives):
        dcg = 0.0
        for idx, source_id in enumerate(predicted[:k], start=1):
            if source_id == positive:
                dcg = 1.0 / math.log2(idx + 1)
                break
        idcg = 1.0
        total += dcg / idcg
    return total / len(predictions)


def attribution_coverage(contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    attributed = sum(1 for context in contexts if "[source:" in context.lower())
    return attributed / len(contexts)


def core_self_ratio(candidate_stances: list[list[str]]) -> float:
    total = 0
    core = 0
    for stances in candidate_stances:
        for stance in stances:
            total += 1
            if stance == "core_self":
                core += 1
    if total == 0:
        return 0.0
    return core / total


def empty_context_rate(contexts: list[str]) -> float:
    if not contexts:
        return 1.0
    empty = sum(1 for context in contexts if not context.strip())
    return empty / len(contexts)


def ideology_consistency_score(texts: list[str]) -> float:
    if not texts:
        return 0.0
    matched = 0
    for text in texts:
        tokens = set(re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", text.lower()))
        if tokens & IDEOLOGY_TERMS:
            matched += 1
    return matched / len(texts)


def citation_hallucination_rate(analyses: list[str], contexts: list[str]) -> float:
    pairs = list(zip(analyses, contexts))
    if not pairs:
        return 0.0

    checked = 0
    hallucinated = 0
    for analysis, context in pairs:
        quotes = re.findall(r"[\"«](.*?)[\"»]", analysis)
        for quote in quotes:
            normalized_quote = quote.strip().lower()
            if len(normalized_quote) < 8:
                continue
            checked += 1
            if normalized_quote not in context.lower():
                hallucinated += 1
    if checked == 0:
        return 0.0
    return hallucinated / checked


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


@dataclass(frozen=True)
class QualityReport:
    recall_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    attribution_coverage: float
    core_self_ratio: float
    empty_context_rate: float
    ideology_consistency: float
    citation_hallucination_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float


def build_quality_report(
    *,
    predictions: list[list[str]],
    positives: list[str],
    contexts: list[str],
    candidate_stances: list[list[str]],
    analyses: list[str],
    latencies_ms: list[float],
) -> QualityReport:
    return QualityReport(
        recall_at_5=recall_at_k(predictions=predictions, positives=positives, k=5),
        mrr_at_10=mrr_at_k(predictions=predictions, positives=positives, k=10),
        ndcg_at_10=ndcg_at_k(predictions=predictions, positives=positives, k=10),
        attribution_coverage=attribution_coverage(contexts=contexts),
        core_self_ratio=core_self_ratio(candidate_stances=candidate_stances),
        empty_context_rate=empty_context_rate(contexts=contexts),
        ideology_consistency=ideology_consistency_score(texts=analyses),
        citation_hallucination_rate=citation_hallucination_rate(
            analyses=analyses, contexts=contexts
        ),
        latency_p50_ms=percentile(values=latencies_ms, p=0.5),
        latency_p95_ms=percentile(values=latencies_ms, p=0.95),
        latency_mean_ms=statistics.mean(latencies_ms) if latencies_ms else 0.0,
    )
