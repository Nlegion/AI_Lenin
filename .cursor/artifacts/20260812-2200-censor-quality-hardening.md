# 20260812-2200 — Censor quality hardening implementation

## Plan
Implemented P0–P5 from censor-quality-hardening plan.

## Result
### P0 Live censor parity
- Added `scripts/_live_news_qa_censor.py` (PreRagCensor build + gate)
- Wired into `run_live_news_qa_batch.py` and 24h path
- Parity: hard_block/skip reject; review → yellow generation
- Disabled unknown soft-pass / unknown_topic_forward for live QA
- Cache parity for context_hints / needs_yellow_warning
- Fail-safe when production censor missing

### P1 Output scrub
- Hardened `output_artifacts.py` (multi-stance, empty slots, cite debris, evidence base)
- `final_public_scrub` after post-guard on both pipeline paths
- Gate threshold for `multi_stance_echo_rate`

### P2 Depth
- Raised answer cap 1000 → 1800
- Adaptive context via `classify_primary`
- Publishability gate for structure/hold/error placeholders

### P3 Degradation
- Typed errors + circuit breaker + non-publishable template degrade
- Precedence documented in `degrade_policy.py`

### P4/P5 Architecture/config
- Extracted `news_item_pipeline.py`
- Per-item DB session isolation
- generation.yaml SoT via `runtime_knobs.py` + `docs/config_ownership.md`

## Verification
- Full pytest: 274 passed, 1 skipped (CLI default fixed)
- Targeted suites for censor/scrub/degrade/processor all green
