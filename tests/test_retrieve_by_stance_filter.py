"""Unit tests for retrieve_by_stance filter behavior."""

from __future__ import annotations


from src.core.retrieval.arbiter import RetrievalCandidate
from src.core.retrieval.stance_retrieve import dedupe_by_chunk_id, merge_slot_candidates, stance_filter


def test_stance_filter_match_any():
    query_filter = stance_filter(stance_types=["core_self"])
    assert query_filter.must
    condition = query_filter.must[0]
    assert condition.key == "stance_type"


def test_merge_slot_candidates_no_cross_stance_boost_needed():
    candidates = [
        RetrievalCandidate(
            source_id="s1",
            chunk_id="c1",
            text="a",
            stance_type="core_self",
            source_path="p1",
            retriever="dense",
            rank=1,
            score=0.9,
        ),
        RetrievalCandidate(
            source_id="s2",
            chunk_id="c2",
            text="b",
            stance_type="core_self",
            source_path="p2",
            retriever="sparse",
            rank=1,
            score=0.8,
        ),
    ]
    merged = merge_slot_candidates(
        candidates=candidates,
        retriever_weights={"dense": 1.0, "sparse": 0.8},
        rrf_k=60,
        limit=2,
    )
    assert {item.chunk_id for item in merged} == {"c1", "c2"}


def test_dedupe_by_chunk_id():
    items = [
        RetrievalCandidate("s", "c1", "t", "core_self", "p", "dense", 1, 1.0),
        RetrievalCandidate("s", "c1", "t2", "core_self", "p", "sparse", 1, 0.5),
    ]
    deduped = dedupe_by_chunk_id(items)
    assert len(deduped) == 1


def test_retrieve_by_stance_docstring_contract():
    from src.core.retrieval.qdrant_retrieval_provider import QdrantRetrievalProvider

    doc = QdrantRetrievalProvider.retrieve_by_stance.__doc__ or ""
    assert "Never returns partial results" in doc
