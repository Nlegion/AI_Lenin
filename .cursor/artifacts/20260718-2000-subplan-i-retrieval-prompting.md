# Subplan I Retrieval and Prompting Execution Report

## Plan
- Implement retrieval policy and prompting layer for personality profile:
  - query rewriting and decomposition,
  - optional HyDE path,
  - arbitration-ready fusion helpers,
  - Qdrant retrieval provider integration into runtime analyzer.
- Validate sandbox run and full project gates.

## Assumptions
- Retrieval provider should be optional and safely fallback to legacy Chroma RAG.
- HyDE in this stage is deterministic query augmentation for low-latency local execution.
- Arbitration logic is implemented as composable helpers and provider ranking flow.

## Questions
- None.

## Result
- Added query transformation module:
  - `src/core/retrieval/query_transform.py`
  - philosophical rewrite, factual/evaluative decomposition, HyDE query builder.
- Added arbiter helpers:
  - `src/core/retrieval/arbiter.py`
  - weighted RRF, stance boosts, core-self enforcement, alpha rerank blend.
- Added Qdrant retrieval provider:
  - `src/core/retrieval/qdrant_retrieval_provider.py`
  - uses dense+sparse+ontology candidates,
  - applies fusion and stance boosts,
  - renders provenance-rich context blocks for prompting.
- Integrated provider into runtime analyzer:
  - `src/core/lenin_analyzer.py`
  - loads `config/retrieval_pipeline.yaml`,
  - tries Qdrant provider first, falls back to legacy `rag_system` on failure.
- Added runtime config:
  - `config/retrieval_pipeline.yaml`
- Extended sparse encoder and ingestion state:
  - `src/core/vector/bm25_sparse.py` now supports save/load state,
  - `src/core/vector/qdrant_ingestion.py`, `scripts/build_qdrant_index.py`, `config/qdrant_ingestion.yaml` updated to persist sparse state.
- Added sandbox orchestrator:
  - `scripts/run_retrieval_sandbox.py`
  - `config/retrieval_sandbox.yaml`

Tests added/updated:
- `tests/test_query_transform.py`
- `tests/test_retrieval_arbiter.py`
- `tests/test_retrieval_sandbox_metrics.py`
- `tests/test_qdrant_ingestion_pipeline.py`

Sandbox snapshot (`30` queries):
- Best mode: `dense`
- `dense` Recall@5: `0.0333`
- `hybrid` Recall@5: `0.0333`
- `hybrid_onto` Recall@5: `0.0333`
- `hyde_hybrid` Recall@5: `0.0333`

## Verification
- Executed commands:
  - `.\\.venv\\Scripts\\python.exe scripts/build_qdrant_index.py --config config/qdrant_ingestion.yaml --chunks-tsv .cursor/artifacts/chunks/chunk_dataset_v2.tsv --limit 2000 --stats-json .cursor/artifacts/qdrant/ingestion_stats.json --summary-md .cursor/artifacts/qdrant/ingestion_summary.md`
  - `.\\.venv\\Scripts\\python.exe scripts/run_retrieval_sandbox.py --config config/retrieval_sandbox.yaml --out-json .cursor/artifacts/sandbox/retrieval_sandbox_results.json --out-md .cursor/artifacts/sandbox/retrieval_sandbox_summary.md --max-queries 30`
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_query_transform.py tests/test_retrieval_arbiter.py tests/test_retrieval_sandbox_metrics.py tests/test_qdrant_ingestion_pipeline.py -q`
  - `.\\.venv\\Scripts\\python.exe scripts/run_subplan_gates.py`
- Gate result:
  - `alembic upgrade head`: PASS
  - `pytest tests -q`: PASS (`25 passed`)

## Reproducibility
- Manifest: `.cursor/artifacts/manifests/20260718-subplan-i.json`
