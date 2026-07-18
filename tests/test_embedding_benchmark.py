from src.core.embeddings.benchmark import (
    BenchmarkResult,
    choose_best_model,
    compute_recall_at_k,
    cosine_similarity,
)


def test_cosine_similarity_handles_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_compute_recall_at_k():
    query_embeddings = [[1.0, 0.0], [0.0, 1.0]]
    document_embeddings = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    positives = [0, 1]
    recall = compute_recall_at_k(
        query_embeddings=query_embeddings,
        document_embeddings=document_embeddings,
        positives=positives,
        k=1,
    )
    assert recall == 1.0


def test_choose_best_model_and_finetune_decision():
    results = [
        BenchmarkResult(
            model_name="model-a",
            recall_at_5=0.82,
            mean_latency_ms=12.0,
            ram_delta_mb=10.0,
            vram_peak_mb=None,
            status="ok",
        ),
        BenchmarkResult(
            model_name="model-b",
            recall_at_5=0.88,
            mean_latency_ms=20.0,
            ram_delta_mb=12.0,
            vram_peak_mb=None,
            status="ok",
        ),
    ]
    winner, should_fine_tune = choose_best_model(results=results, min_recall_at_5=0.85)
    assert winner is not None
    assert winner.model_name == "model-b"
    assert should_fine_tune is False
