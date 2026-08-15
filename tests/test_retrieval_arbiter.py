from src.core.retrieval.arbiter import (
    RetrievalCandidate,
    apply_stance_boost,
    enforce_core_self_presence,
    rerank_with_alpha,
    weighted_rrf,
)


def _candidate(
    chunk_id: str, retriever: str, rank: int, stance: str
) -> RetrievalCandidate:
    return RetrievalCandidate(
        source_id=f"src_{chunk_id}",
        chunk_id=chunk_id,
        text="text",
        stance_type=stance,
        source_path="path",
        retriever=retriever,
        rank=rank,
        score=1.0,
    )


def test_weighted_rrf_and_boost():
    candidates = [
        _candidate("a", "dense", 1, "core_self"),
        _candidate("b", "dense", 2, "contextual"),
    ]
    scores = weighted_rrf(
        candidates=candidates, retriever_weights={"dense": 1.0, "sparse": 0.8}, rrf_k=60
    )
    boosted = apply_stance_boost(
        scores=scores,
        chunk_to_stance={"a": "core_self", "b": "contextual"},
        source_boosts={"core_self": 1.5, "contextual": 0.9},
    )
    assert boosted["a"] > 0
    assert boosted["a"] > boosted["b"]


def test_enforce_core_self_presence_and_reranker_blend():
    ordered = [f"id_{index}" for index in range(12)] + ["core_id"]
    stances = {chunk_id: "contextual" for chunk_id in ordered}
    stances["core_id"] = "core_self"
    reordered = enforce_core_self_presence(
        ordered_chunk_ids=ordered, chunk_to_stance=stances
    )
    assert reordered[0] == "core_id"

    merged = {"x": 0.4, "y": 0.2}
    reranked = {"x": 0.0, "y": 1.0}
    final_scores = rerank_with_alpha(
        merged_scores=merged, reranker_scores=reranked, alpha=0.3
    )
    assert final_scores["y"] > final_scores["x"]
