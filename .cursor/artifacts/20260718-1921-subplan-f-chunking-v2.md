# Subplan F Chunking V2 Execution Report

## Plan
- Implement chunking v2 with philosophical structure awareness and stable metadata-rich chunks.
- Enforce token windows (`256-512`) with overlap (~10%).
- Validate thesis-boundary quality target and export machine-readable chunk dataset.
- Run mandatory gates and create reproducibility manifest.

## Assumptions
- Input corpus comes from Subplan E cleaned outputs in `.cursor/artifacts/cleaned_corpus`.
- Source provenance comes from Subplan B source registry.
- Boundary quality is evaluated by chunk boundary safety ratio and token-window compliance.

## Questions
- None.

## Result
- Added chunking config loader:
  - `src/core/preprocessing/chunking_config.py`
- Added chunking v2 engine:
  - `src/core/preprocessing/chunker_v2.py`
  - hierarchy-aware metadata (`chapter`, `section`, `paragraph_index`, `thesis_index`),
  - stable `chunk_id`,
  - token-window slicing with overlap and deterministic boundaries.
- Added quality checks:
  - `src/core/preprocessing/chunking_quality.py`
  - bad-boundary ratio + token-window compliance ratio.
- Added chunk dataset builder:
  - `scripts/build_chunk_dataset_v2.py`
  - outputs chunk TSV + QA JSON + summary markdown.
- Added config:
  - `config/chunking_rules.yaml`
- Added tests:
  - `tests/test_chunking_v2.py`

Generated artifacts:
- `.cursor/artifacts/chunks/chunk_dataset_v2.tsv`
- `.cursor/artifacts/chunks/chunking_summary.md`
- `.cursor/artifacts/chunks/chunking_qa.json`

Quality snapshot:
- Source documents processed: `148`
- Total chunks: `51718`
- Mean tokens/chunk: `511.34`
- Token window compliance ratio: `0.9988`
- Bad boundary ratio: `0.0000`
- Boundary target (`<= 0.2000`) passed: `yes`

## Verification
- Executed commands:
  - `.\\.venv\\Scripts\\python.exe scripts/build_chunk_dataset_v2.py --registry .cursor/artifacts/registries/source_registry.tsv --cleaned-root .cursor/artifacts/cleaned_corpus --config config/chunking_rules.yaml --chunks-output .cursor/artifacts/chunks/chunk_dataset_v2.tsv --summary-output .cursor/artifacts/chunks/chunking_summary.md --qa-output .cursor/artifacts/chunks/chunking_qa.json`
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_chunking_v2.py -q`
  - `.\\.venv\\Scripts\\python.exe scripts/run_subplan_gates.py`
- Gate result:
  - `alembic upgrade head`: PASS
  - `pytest tests -q`: PASS (`14 passed`)

## Reproducibility
- Manifest: `.cursor/artifacts/manifests/20260718-subplan-f.json`
