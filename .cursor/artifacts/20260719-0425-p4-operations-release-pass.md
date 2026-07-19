# P4 Operational Hardening and Release-pass

## Implemented
- Added unified entrypoint package:
  - `ai_lenin/__init__.py`
  - `ai_lenin/entrypoint.py`
  - launch pattern: `python -m ai_lenin.entrypoint`
- Added bootstrap script and dev dependency profile:
  - `requirements-dev.txt`
  - `scripts/bootstrap_dev_env.py`
- Added release-pass script:
  - `scripts/release_pass.py`
  - includes subplan gates + NewsGuard safety regression.
- Expanded optional gate support in `scripts/run_subplan_gates.py` to include `vulture`.
- Added public policy validator:
  - `scripts/validate_public_news_policy.py`

## Validation
- `python scripts/bootstrap_dev_env.py --skip-runtime` -> `bootstrap_complete`
- `python -c "import ai_lenin.entrypoint"` -> `entrypoint_import_ok`
- `python scripts/evaluate_news_guard.py ...` -> provocative blocked/quarantined `50/50`
- `python scripts/validate_public_news_policy.py --public-mode` -> `public_policy_ok`

## Release-pass Status
- `scripts/release_pass.py` currently returns **FAIL** due existing repository-wide lint debt in legacy test files under `tests/` (e.g., `tests/test_news_fetcher.py`).
- Core runtime gates (`alembic`, `pytest tests -q`) pass.
- Action required for full green release-pass:
  - clean existing `ruff` violations in test/legacy files or
  - formalize and approve reduced lint scope for production-critical modules.
