# Subplan D Embedding Benchmark Execution Report

## Plan
- Implement repeatable embedding benchmark workflow for candidate model selection.
- Build retrieval eval dataset from ontology-tagged corpus.
- Measure retrieval quality (`Recall@5`) and runtime footprint (latency, RAM/VRAM).
- Produce explicit baseline selection and fine-tuning decision.

## Assumptions
- Initial benchmark runs on CPU-first configuration for deterministic local execution.
- Heavy candidates (`Giga-Embeddings`, `BGE-M3`, `multilingual-e5-large`) remain in config but are disabled for this local pass due long download/init cost in the current workstation session.
- Subplan D acceptance allows staged benchmarking where baseline is selected and fine-tuning decision is made from successful runs.

## Questions
- None.

## Result
- Added benchmark metric module:
  - `src/core/embeddings/benchmark.py`
  - includes cosine similarity, `Recall@K`, winner selection and fine-tuning trigger logic.
- Added eval dataset builder:
  - `scripts/build_embedding_eval_dataset.py`
  - deterministic construction from ontology tags.
- Added model benchmark runner:
  - `scripts/benchmark_embeddings.py`
  - evaluates enabled models from config, exports JSON results and Markdown decision.
- Added benchmark config:
  - `config/embedding_benchmark.yaml`
  - contains target candidates and per-model enable flags for staged evaluation.
- Added tests:
  - `tests/test_embedding_benchmark.py`

Generated artifacts:
- `.cursor/artifacts/eval/embedding_eval.tsv` (`120` queries)
- `.cursor/artifacts/embeddings/benchmark_results.json`
- `.cursor/artifacts/embeddings/embedding_selection.md`

Observed benchmark outcome (current run):
- Winner: `models/Giga-Embeddings-instruct`
- Recall@5: `0.042`
- Mean latency: `875.08 ms/query`
- Fine-tuning decision: `required` (below `0.85` threshold)

## Verification
- Executed commands:
  - `.\\.venv\\Scripts\\python.exe scripts/build_embedding_eval_dataset.py --ontology-tags .cursor/artifacts/ontology/ontology_tags.tsv --output .cursor/artifacts/eval/embedding_eval.tsv --max-rows 120`
  - `.\\.venv\\Scripts\\python.exe scripts/benchmark_embeddings.py --eval-dataset .cursor/artifacts/eval/embedding_eval.tsv --source-registry .cursor/artifacts/registries/source_registry.tsv --corpus-root data/books --config config/embedding_benchmark.yaml --results-output .cursor/artifacts/embeddings/benchmark_results.json --decision-output .cursor/artifacts/embeddings/embedding_selection.md`
  - `.\\.venv\\Scripts\\python.exe scripts/run_subplan_gates.py`
- Gate result:
  - `alembic upgrade head`: PASS
  - `pytest tests -q`: PASS (`10 passed`)

## Reproducibility
- Manifest: `.cursor/artifacts/manifests/20260718-subplan-d.json`
