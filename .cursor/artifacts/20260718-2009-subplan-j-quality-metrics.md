# Subplan J Quality Metrics Execution Report

## Plan
- Implement quality evaluation metrics for the rebuilt RAG pipeline.
- Run retrieval-based evaluation with explicit thresholds:
  - Recall@5, MRR@10, nDCG@10,
  - attribution coverage,
  - core_self_ratio,
  - empty context rate,
  - ideology consistency,
  - citation hallucination rate,
  - latency p50/p95.
- Extend cycle checks with `ruff`, `bandit`, and `vulture`.

## Assumptions
- Evaluation uses current local smoke index and local retrieval provider configuration.
- Citation hallucination is measured against generated quote snippets derived from retrieved context.
- Thresholds are fixed by `config/quality_thresholds.yaml`.

## Questions
- None.

## Result
- Added metrics module:
  - `src/core/evaluation/rag_quality_metrics.py`
- Added quality evaluation runner:
  - `scripts/evaluate_rag_quality.py`
- Added thresholds config:
  - `config/quality_thresholds.yaml`
- Added tests:
  - `tests/test_rag_quality_metrics.py`

Evaluation artifacts:
- `.cursor/artifacts/evaluation/rag_quality_summary.md`
- `.cursor/artifacts/evaluation/rag_quality_metrics.json` (local JSON)

Observed evaluation snapshot (`30` queries):
- Recall@5: `0.0333` (target `0.85`) -> fail
- MRR@10: `0.0067`
- nDCG@10: `0.0129`
- Attribution coverage: `1.0000` -> pass
- Core self ratio: `0.0000` (target `0.70`) -> fail
- Empty context rate: `0.0000` -> pass
- Ideology consistency: `0.0667` (target `0.70`) -> fail
- Citation hallucination rate: `0.0000` (max `0.05`) -> pass
- Latency p50/p95: `357.81 / 376.18 ms`

## Verification
- Executed commands:
  - `.\\.venv\\Scripts\\python.exe scripts/evaluate_rag_quality.py --retrieval-config config/retrieval_pipeline.yaml --thresholds-config config/quality_thresholds.yaml --eval-dataset .cursor/artifacts/eval/embedding_eval.tsv --output-json .cursor/artifacts/evaluation/rag_quality_metrics.json --output-md .cursor/artifacts/evaluation/rag_quality_summary.md --max-queries 30`
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_rag_quality_metrics.py tests/test_query_transform.py tests/test_retrieval_arbiter.py tests/test_retrieval_sandbox_metrics.py tests/test_qdrant_ingestion_pipeline.py -q`
  - `.\\.venv\\Scripts\\python.exe scripts/run_subplan_gates.py`
  - `.\\.venv\\Scripts\\python.exe -m ruff check src/core/evaluation/rag_quality_metrics.py src/core/retrieval/query_transform.py src/core/retrieval/arbiter.py src/core/retrieval/qdrant_retrieval_provider.py src/core/lenin_analyzer.py src/core/vector/bm25_sparse.py src/core/vector/qdrant_ingestion.py scripts/evaluate_rag_quality.py scripts/run_retrieval_sandbox.py scripts/build_qdrant_index.py tests/test_rag_quality_metrics.py tests/test_query_transform.py tests/test_retrieval_arbiter.py`
  - `.\\.venv\\Scripts\\python.exe -m bandit -q -r src/core/evaluation src/core/retrieval src/core/vector scripts/evaluate_rag_quality.py scripts/run_retrieval_sandbox.py scripts/build_qdrant_index.py`
  - `.\\.venv\\Scripts\\python.exe -m vulture src/core/evaluation src/core/retrieval src/core/vector scripts/evaluate_rag_quality.py scripts/run_retrieval_sandbox.py scripts/build_qdrant_index.py tests/test_rag_quality_metrics.py tests/test_query_transform.py tests/test_retrieval_arbiter.py --min-confidence 100`

Gate result:
- `alembic upgrade head`: PASS
- `pytest tests -q`: PASS (`28 passed`)
- `ruff`: PASS
- `bandit`: PASS
- `vulture`: PASS

## Reproducibility
- Manifest: `.cursor/artifacts/manifests/20260718-subplan-j.json`
