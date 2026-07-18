# Subplan H Retrieval Sandbox Execution Report

## Plan
- Implement retrieval sandbox for iterative experiments across:
  - dense only,
  - dense + sparse hybrid,
  - dense + sparse + ontology hybrid,
  - HyDE-augmented hybrid.
- Add reproducible fusion/metric utilities and run experiments on local Qdrant smoke index.
- Validate with tests and full project gates.

## Assumptions
- Sandbox experiments run on local embedded Qdrant collection from Subplan G.
- Sparse encoder state is persisted from ingestion for query-time sparse encoding.
- HyDE is implemented as lightweight query transformation in this phase (no separate generation model yet).

## Questions
- None.

## Result
- Extended BM25 encoder state management:
  - `src/core/vector/bm25_sparse.py`
  - added `save()` / `load()` for reusable sparse vocabulary+idf state.
- Extended Qdrant ingestion config/pipeline:
  - `src/core/vector/qdrant_ingestion.py`
  - persisted sparse state artifact for retrieval-time sparse queries.
  - `scripts/build_qdrant_index.py`, `config/qdrant_ingestion.yaml` updated accordingly.
- Added retrieval sandbox metrics/fusion helpers:
  - `src/core/vector/sandbox_metrics.py`
  - weighted RRF, source stance boosts, Recall@K.
- Added sandbox experiment runner:
  - `scripts/run_retrieval_sandbox.py`
  - executes modes: `dense`, `hybrid`, `hybrid_onto`, `hyde_hybrid`.
- Added sandbox config:
  - `config/retrieval_sandbox.yaml`
- Added tests:
  - `tests/test_retrieval_sandbox_metrics.py`
  - updated `tests/test_qdrant_ingestion_pipeline.py` for sparse-state roundtrip.

Sandbox artifacts:
- `.cursor/artifacts/sandbox/retrieval_sandbox_summary.md`
- `.cursor/artifacts/sandbox/retrieval_sandbox_results.json` (local JSON, ignored by git)

Sandbox snapshot (`30` queries):
- Best mode: `dense`
- `dense` Recall@5: `0.0333`
- `hybrid` Recall@5: `0.0333`
- `hybrid_onto` Recall@5: `0.0333`
- `hyde_hybrid` Recall@5: `0.0333`

## Verification
- Executed commands:
  - `.\\.venv\\Scripts\\python.exe scripts/build_qdrant_index.py --config config/qdrant_ingestion.yaml --chunks-tsv .cursor/artifacts/chunks/chunk_dataset_v2.tsv --limit 1500 --stats-json .cursor/artifacts/qdrant/ingestion_stats.json --summary-md .cursor/artifacts/qdrant/ingestion_summary.md`
  - `.\\.venv\\Scripts\\python.exe scripts/run_retrieval_sandbox.py --config config/retrieval_sandbox.yaml --out-json .cursor/artifacts/sandbox/retrieval_sandbox_results.json --out-md .cursor/artifacts/sandbox/retrieval_sandbox_summary.md --max-queries 30`
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_retrieval_sandbox_metrics.py tests/test_qdrant_ingestion_pipeline.py -q`
  - `.\\.venv\\Scripts\\python.exe scripts/run_subplan_gates.py`
- Gate result:
  - `alembic upgrade head`: PASS
  - `pytest tests -q`: PASS (`20 passed`)

## Reproducibility
- Manifest: `.cursor/artifacts/manifests/20260718-subplan-h.json`
