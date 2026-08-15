from src.core.vector.sandbox_metrics import (
    apply_source_boost,
    recall_at_k,
    reciprocal_rank_fusion,
)


def test_reciprocal_rank_fusion_scores_ordering():
    rank_lists = {
        "dense": ["a", "b", "c"],
        "sparse": ["b", "a", "d"],
    }
    weights = {"dense": 1.0, "sparse": 1.0}
    scores = reciprocal_rank_fusion(
        rank_lists=rank_lists, retriever_weights=weights, k=60
    )
    assert scores["a"] > 0
    assert scores["b"] > scores["c"]


def test_apply_source_boost_changes_priority():
    scores = {"a": 1.0, "b": 1.0}
    source_stance = {"a": "core_self", "b": "contextual"}
    boosts = {"core_self": 1.5, "contextual": 0.9}
    boosted = apply_source_boost(
        scores=scores, source_stance=source_stance, boosts=boosts
    )
    assert boosted["a"] > boosted["b"]


def test_recall_at_k_computation():
    predictions = [["a", "b"], ["x", "y"], ["k", "z"]]
    positives = ["b", "q", "k"]
    assert recall_at_k(predictions=predictions, positives=positives, k=2) == (2 / 3)
