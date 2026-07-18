from src.core.evaluation.rag_quality_metrics import (
    attribution_coverage,
    build_quality_report,
    citation_hallucination_rate,
    core_self_ratio,
    empty_context_rate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_rank_metrics():
    predictions = [["a", "b", "c"], ["x", "q", "z"]]
    positives = ["b", "q"]
    assert recall_at_k(predictions=predictions, positives=positives, k=2) == 1.0
    assert mrr_at_k(predictions=predictions, positives=positives, k=3) > 0.0
    assert ndcg_at_k(predictions=predictions, positives=positives, k=3) > 0.0


def test_context_and_stance_metrics():
    contexts = ["[source: a] text", ""]
    stances = [["core_self", "contextual"], ["contextual"]]
    assert attribution_coverage(contexts=contexts) == 0.5
    assert empty_context_rate(contexts=contexts) == 0.5
    assert core_self_ratio(candidate_stances=stances) == (1 / 3)


def test_citation_hallucination_and_report():
    analyses = ['Как я писал: "прибавочная стоимость создается трудом"']
    contexts = ["[source: x] ... прибавочная стоимость создается трудом ..."]
    assert citation_hallucination_rate(analyses=analyses, contexts=contexts) == 0.0

    report = build_quality_report(
        predictions=[["x"]],
        positives=["x"],
        contexts=contexts,
        candidate_stances=[["core_self"]],
        analyses=analyses,
        latencies_ms=[10.0, 20.0, 30.0],
    )
    assert report.recall_at_5 == 1.0
    assert report.latency_p95_ms >= report.latency_p50_ms
