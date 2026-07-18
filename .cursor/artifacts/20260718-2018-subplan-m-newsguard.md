# Subplan M — NewsGuard Safety Layer

## Plan
- Scope: implement a two-tier safety module (`NewsGate` + `NewsGuard`) with YAML policy, integrate it into runtime processing/publishing, and add tests plus evaluation artifacts.
- Acceptance criteria:
  - Input gate supports `allow/deny/quarantine` decisions with reason codes.
  - Output guard blocks unsafe patterns and enforces mandatory disclaimer.
  - Safety behavior is configurable in `config/news_guard.yaml`.
  - Provocative test batch demonstrates stable blocking/quarantine behavior.
- Planned commands:
  - `python -m pytest tests/test_news_guard.py tests/test_news_processor.py -q`
  - `python scripts/evaluate_news_guard.py --config config/news_guard.yaml`
  - `python scripts/run_subplan_gates.py --run-optional`
  - `python -m bandit -q -r src/core/safety src/core/processor.py scripts/evaluate_news_guard.py`
  - `python -m vulture ... --min-confidence 100`

## Assumptions
- Existing moderation in `NewsClassifier` remains useful as a topic-level filter and is complemented (not replaced) by NewsGuard policy checks.
- `quarantine` on input is treated as skip/no-generation in current runtime loop.

## Questions
- None.

## Result
- Changes implemented:
  - Added `src/core/safety/news_guard.py` with typed config models, input gating, output moderation, regex blocking, and disclaimer injection.
  - Added runtime policy file `config/news_guard.yaml`.
  - Integrated guard calls into `src/core/processor.py` for:
    - pre-generation decision (`deny/quarantine` => skip),
    - post-generation moderation,
    - pre-publish moderation to guarantee disclaimer enforcement.
  - Removed duplicate legacy `process_single_news` definition in `processor.py` to avoid shadowing and lint instability.
  - Added `scripts/evaluate_news_guard.py` to generate reproducible safety snapshots (`.cursor/artifacts/safety/*.json|md`).
  - Added tests `tests/test_news_guard.py` and package init `src/core/safety/__init__.py`.
- Files changed:
  - `src/core/processor.py`
  - `src/core/safety/news_guard.py`
  - `src/core/safety/__init__.py`
  - `config/news_guard.yaml`
  - `scripts/evaluate_news_guard.py`
  - `tests/test_news_guard.py`
  - `.cursor/artifacts/safety/news_guard_eval.json`
  - `.cursor/artifacts/safety/news_guard_eval_summary.md`
- Risks / follow-ups:
  - Current policy is keyword/regex driven; if false positives appear, tune allow/quarantine lists and add stricter semantic classifier in a follow-up iteration.

## Verification
- Executed commands:
  - `python -m alembic upgrade head` (via gate script) — pass.
  - `python -m pytest tests -q` (via gate script) — pass.
  - `python -m pytest tests/test_news_guard.py tests/test_news_processor.py -q` — pass.
  - `python -m ruff check` on changed files — pass.
  - `python -m bandit -q -r src/core/safety src/core/processor.py scripts/evaluate_news_guard.py` — pass.
  - `python -m vulture ... --min-confidence 100` on changed files — pass.
  - `python scripts/evaluate_news_guard.py` — generated safety evaluation snapshot.
- Outcome:
  - Provocative cases blocked/quarantined: `50/50`.
  - Allowed cases approved by input gate: `3/5`.
  - Disclaimer present in provocative output checks: `50/50`.

## Reproducibility
- Manifest file: `.cursor/artifacts/manifests/20260718-subplan-m.json`
- Config/model hashes: captured in manifest.
