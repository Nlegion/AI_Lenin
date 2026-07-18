# Subplan K — Retrieval Migration Compatibility

## Plan
- Introduce a stable `RetrievalProvider` abstraction used by runtime generation.
- Add migration modes for controlled cutover from legacy Chroma to new Qdrant retrieval.
- Run A/B shadow checks with parity logging and produce migration readiness artifacts.

## Assumptions
- Current production fallback (`EnhancedRAGSystem`) remains available as Chroma legacy source.
- Cutover should keep generation behavior stable by returning primary provider context while shadowing secondary provider.

## Questions
- None.

## Result
- Changes implemented:
  - Added provider contract `src/core/retrieval/base_provider.py` with normalized `RetrievalResult`.
  - Added `src/core/retrieval/chroma_retrieval_provider.py` wrapping legacy Chroma retrieval.
  - Extended `src/core/retrieval/qdrant_retrieval_provider.py` with `retrieve_context(...)` method for contract parity.
  - Added migration orchestrator `src/core/retrieval/migration_provider.py`:
    - modes: `qdrant_only`, `chroma_only`, `ab_shadow`,
    - parity computation (`shared_ratio`) and threshold warning,
    - JSONL audit logging with hashed query payload.
  - Added `src/core/retrieval/provider_factory.py` with typed YAML config loading and provider wiring.
  - Updated `src/core/lenin_analyzer.py` to initialize provider via factory and consume normalized retrieval result.
  - Updated `config/retrieval_pipeline.yaml` with migration section:
    - `mode`,
    - `parity_min_shared_ratio`,
    - `audit_log_path`,
    - `chroma_top_k`.
  - Added parity runner `scripts/run_retrieval_ab_check.py`.
  - Added tests:
    - `tests/test_migration_retrieval_provider.py`,
    - `tests/test_retrieval_provider_factory.py`.
- Files changed:
  - `config/retrieval_pipeline.yaml`
  - `src/core/lenin_analyzer.py`
  - `src/core/retrieval/__init__.py`
  - `src/core/retrieval/base_provider.py`
  - `src/core/retrieval/chroma_retrieval_provider.py`
  - `src/core/retrieval/migration_provider.py`
  - `src/core/retrieval/provider_factory.py`
  - `src/core/retrieval/qdrant_retrieval_provider.py`
  - `scripts/run_retrieval_ab_check.py`
  - `tests/test_migration_retrieval_provider.py`
  - `tests/test_retrieval_provider_factory.py`
  - `.cursor/artifacts/retrieval/retrieval_ab_summary.md`
  - `.cursor/artifacts/retrieval/retrieval_ab_audit.jsonl`
- Risks / follow-ups:
  - Shadow parity average is currently low, so immediate hard cutover is not advised without tuning retrieval policy/fusion weights.

## Verification
- Executed commands:
  - `python -m alembic upgrade head`
  - `python -m pytest tests -q`
  - `python -m pytest tests/test_migration_retrieval_provider.py tests/test_retrieval_provider_factory.py tests/test_news_processor.py -q`
  - `python -m ruff check <changed-files>`
  - `python -m bandit -q -r src/core/retrieval src/core/lenin_analyzer.py scripts/run_retrieval_ab_check.py`
  - `python -m vulture <changed-files> --min-confidence 100`
  - `python scripts/run_retrieval_ab_check.py --config config/retrieval_pipeline.yaml`
- Outcome:
  - full test suite passed,
  - static/security/dead-code checks passed for Subplan K scope,
  - A/B shadow snapshot generated:
    - non-empty context rate: `1.000`,
    - average shared ratio: `0.05926`,
    - threshold breaches logged as warnings.

## Reproducibility
- Manifest file: `.cursor/artifacts/manifests/20260718-subplan-k.json`
- Config/model hashes: captured in manifest.
