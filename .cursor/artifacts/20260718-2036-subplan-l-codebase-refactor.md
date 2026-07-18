# Subplan L — Codebase Refactor and Release Readiness

## Plan
- Decompose overloaded runtime logic in analyzer retrieval path.
- Centralize analysis/runtime literals into settings constants.
- Add controlled legacy RAG inventory workflow for non-destructive archival planning.
- Validate with full gates and static/security checks.

## Assumptions
- Full removal of legacy Chroma fallback is deferred until migration parity reaches target.
- Legacy cleanup in this subplan is "managed archive planning" (inventory + policy), not immediate file deletion.

## Questions
- None.

## Result
- Changes implemented:
  - Added `src/core/analysis/context_orchestrator.py` and `src/core/analysis/__init__.py` to isolate retrieval context orchestration from `LeninAnalyzer`.
  - Refactored `src/core/lenin_analyzer.py` to use `AnalysisContextOrchestrator` for provider-first retrieval with legacy fallback.
  - Centralized generation/refusal/runtime defaults in `src/core/settings/analysis_defaults.py`.
  - Reused centralized refusal phrases in `src/core/processor.py`.
  - Added typed legacy registry loader `src/core/settings/legacy_registry.py`.
  - Added managed legacy policy config `config/legacy_rag_components.yaml`.
  - Added inventory script `scripts/build_legacy_rag_inventory.py` generating reproducible legacy cleanup artifacts.
  - Added tests:
    - `tests/test_context_orchestrator.py`
    - `tests/test_legacy_registry.py`
  - Generated legacy inventory artifact:
    - `.cursor/artifacts/legacy/legacy_rag_inventory.md`
- Files changed:
  - `src/core/lenin_analyzer.py`
  - `src/core/processor.py`
  - `src/core/analysis/__init__.py`
  - `src/core/analysis/context_orchestrator.py`
  - `src/core/settings/analysis_defaults.py`
  - `src/core/settings/legacy_registry.py`
  - `config/legacy_rag_components.yaml`
  - `scripts/build_legacy_rag_inventory.py`
  - `tests/test_context_orchestrator.py`
  - `tests/test_legacy_registry.py`
  - `.cursor/artifacts/legacy/legacy_rag_inventory.md`
- Risks / follow-ups:
  - Inventory marks deprecated files as archive candidates; final physical archive/removal should happen after explicit cutover approval.

## Verification
- Executed commands:
  - `python -m alembic upgrade head`
  - `python -m pytest tests -q`
  - `python -m pytest tests/test_context_orchestrator.py tests/test_legacy_registry.py tests/test_news_processor.py -q`
  - `python -m ruff check <changed-files>`
  - `python -m bandit -q -r <changed-scope>`
  - `python -m vulture <changed-files> --min-confidence 100`
  - `python scripts/build_legacy_rag_inventory.py --config config/legacy_rag_components.yaml`
- Outcome:
  - all checks passed for Subplan L scope,
  - full test suite passed,
  - legacy inventory artifact built with 3 tracked components.

## Reproducibility
- Manifest file: `.cursor/artifacts/manifests/20260718-subplan-l.json`
- Config/model hashes: captured in manifest.
