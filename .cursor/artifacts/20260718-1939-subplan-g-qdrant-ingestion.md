# Subplan G Qdrant Ingestion Execution Report

## Plan
- Implement local Qdrant ingestion infrastructure for dense+sparse retrieval.
- Add BM25 sparse encoder and Qdrant upsert pipeline with retries/checkpointing.
- Add cache pre-warming for core/self sources.
- Run smoke ingestion, required tests, and full project gates.

## Assumptions
- For deterministic local execution, dense model is CPU-based (`Giga-Embeddings-instruct`) in this stage.
- Qdrant runs in embedded local mode (`QdrantClient(path=...)`) without Docker dependency.
- Full corpus ingestion can be run later; this stage validates pipeline correctness with smoke volume.

## Questions
- None.

## Result
- Added BM25 sparse encoder:
  - `src/core/vector/bm25_sparse.py`
- Added Qdrant ingestion pipeline:
  - `src/core/vector/qdrant_ingestion.py`
  - collection bootstrap for dense+sparse vectors,
  - batch upsert with retries,
  - checkpoint offset persistence,
  - core-first cache prewarm with fallback.
- Added runner script:
  - `scripts/build_qdrant_index.py`
- Added Qdrant config:
  - `config/qdrant_ingestion.yaml`
- Added tests:
  - `tests/test_qdrant_ingestion_pipeline.py`
- Added runtime dependency:
  - `qdrant-client==1.15.1` in `requirements.txt`.

Smoke ingestion artifacts:
- `.cursor/artifacts/qdrant/ingestion_summary.md`
- `.cursor/artifacts/qdrant/ingestion_stats.json` (local, JSON ignored in git)

Smoke run snapshot (`limit=1000`):
- Rows total: `1000`
- Rows processed: `1000`
- Checkpoint offset: `1000`
- Prewarmed points: `200`

## Verification
- Executed commands:
  - `.\\.venv\\Scripts\\python.exe -m pip install qdrant-client -q`
  - `.\\.venv\\Scripts\\python.exe scripts/build_qdrant_index.py --config config/qdrant_ingestion.yaml --chunks-tsv .cursor/artifacts/chunks/chunk_dataset_v2.tsv --limit 1000 --stats-json .cursor/artifacts/qdrant/ingestion_stats.json --summary-md .cursor/artifacts/qdrant/ingestion_summary.md`
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_qdrant_ingestion_pipeline.py -q`
  - `.\\.venv\\Scripts\\python.exe scripts/run_subplan_gates.py`
- Gate result:
  - `alembic upgrade head`: PASS
  - `pytest tests -q`: PASS (`16 passed`)

## Reproducibility
- Manifest: `.cursor/artifacts/manifests/20260718-subplan-g.json`
