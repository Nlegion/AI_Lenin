# Subplan E Cleaning Rebuild Execution Report

## Plan
- Rebuild corpus cleaning pipeline from scratch using explicit, configurable rules.
- Add QA validation for semantic preservation with strict damage threshold `<2%`.
- Produce cleaned corpus artifacts and quality summary.
- Run mandatory gates and record reproducibility manifest.

## Assumptions
- Input corpus is derived from Subplan B source registry and `data/books`.
- This phase focuses on deterministic cleaning and quality checks, without semantic rewriting.
- Validation is computed on deterministic random sample (`seed=42`).

## Questions
- None.

## Result
- Added cleaning configuration loader:
  - `src/core/preprocessing/cleaning_config.py`
- Added deterministic cleaner:
  - `src/core/preprocessing/text_cleaner.py`
  - content-start detection, line-noise and inline-noise removal, normalization.
- Added quality metrics:
  - `src/core/preprocessing/cleaning_quality.py`
  - paragraph-level semantic damage estimation via token-overlap.
- Added full rebuild script:
  - `scripts/rebuild_clean_corpus.py`
  - processes all registry files, writes cleaned corpus, computes QA JSON and markdown summary.
- Added cleaning policy config:
  - `config/cleaning_rules.yaml`
- Added tests:
  - `tests/test_cleaning_pipeline.py`

Generated artifacts:
- `.cursor/artifacts/cleaned_corpus/...` (rebuilt cleaned texts)
- `.cursor/artifacts/cleaning/cleaning_qa.json`
- `.cursor/artifacts/cleaning/cleaning_summary.md`

QA outcome:
- Processed files: `148`
- Written files: `148`
- Mean size reduction: `1.90%`
- Validation sample size: `25`
- Mean semantic damage ratio: `0.0008`
- Max semantic damage ratio: `0.0190`
- Threshold target: `< 0.0200`
- Threshold passed: `yes`

## Verification
- Executed commands:
  - `.\\.venv\\Scripts\\python.exe scripts/rebuild_clean_corpus.py --registry .cursor/artifacts/registries/source_registry.tsv --corpus-root data/books --config config/cleaning_rules.yaml --cleaned-root .cursor/artifacts/cleaned_corpus --qa-json .cursor/artifacts/cleaning/cleaning_qa.json --summary-md .cursor/artifacts/cleaning/cleaning_summary.md`
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_cleaning_pipeline.py -q`
  - `.\\.venv\\Scripts\\python.exe scripts/run_subplan_gates.py`
- Gate result:
  - `alembic upgrade head`: PASS
  - `pytest tests -q`: PASS (`12 passed`)

## Reproducibility
- Manifest: `.cursor/artifacts/manifests/20260718-subplan-e.json`
