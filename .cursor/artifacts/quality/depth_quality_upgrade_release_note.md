# Depth Quality Upgrade — Release Validation Note

## Scope
- Plan: `depth_quality_upgrade`
- Branch validation date: 2026-08-09
- Focus: containment, metric schema hardening, artifact integrity, depth/grounding guardrails, unknown-topic conflict reporting

## Implemented Phases

### Phase 0 (containment)
- Added config-switchable ethno-hate containment in `pre_rag_censor`:
  - `ethno_hate_containment_enabled` runtime flag
  - new hard-block code: `manual_ethno_hate_containment`
- Added tests for obfuscated ethno-hate phrase and config off-switch.

### Phase 1 (measurement/baseline)
- Extended QA metrics schema to `2.0-depth-gating`.
- Added denominator and cohort fields:
  - `eligible_generated_n`, `llm_attempted_n`, `llm_generated_n`, `llm_final_used_n`, `skipped_llm_n`
  - quote diagnostics (`quote_candidate_found_rate`, `quote_required_rate`, `quote_fulfilled_rate`, etc.)
  - route split (`routing_all`, `routing_generated`)
  - leak split (`chatml_leak_rate`, `orchestrator_label_leak_rate`, `multi_stance_echo_rate`)
- Added quote applicability flags to avoid vacuous pass.

### Phase 2 (safety/integrity)
- Enabled `loop_fix_enabled: true` in `config/quality_postcheck.yaml`.
- Added artifact stripping for:
  - ChatML markers `<|im_start|>/<|im_end|>`
  - repeated `[multi-stance]`
  - user-visible `R1/R2/R3`-style labels
- Added unit test for artifact stripping behavior.

### Phase 2.5 (integration smoke)
- Bounded QA run executed (`limit=20`) with scrubbers active.
- Structural rebuild marker exported (`structure_rebuilt`) for smoke inspection.

### Phase 3 (depth/grounding)
- Prompt contract upgraded to explicit structure:
  - `Факт -> Механизм -> Вывод`
- Added explicit ban of internal orchestration labels/tokens in prompt.
- Added deterministic post-generation structure enforcement (`structure_rebuilt` metadata).
- Added optional baseline ratio gates for latency and answer length in metrics runner.

### Phase 4 (unknown-topic refinement support)
- Added explicit override trace code:
  - `override:unknown_topic_forward_trusted_source`
- Updated `analyze_censorship_run.py` to separate:
  - `intentional_override_rows`
  - `unexpected_conflict_rows`

## Validation Runs

### Tests
- `pytest tests/test_pre_rag_censor.py tests/test_output_artifact_hardening.py tests/test_quality_metrics_denominators.py -q`
- Result: `23 passed`

### Censorship replay gate check
- Replay artifact: `censorship_1h_baseline_replay_postphase.jsonl`
- Gate command: `check_censorship_gates.py`
- Result: **FAILED**  
  - `review_rate` above gate (`0.4146` > `0.20`)
  - `p95_ratio_vs_baseline` above gate (`1.189` > `1.10`)
  - `throughput_ratio_vs_baseline` below gate (`0.014` < `0.8`)

### Bounded live censorship check
- Run: `run_censorship_isolated_24h.py --config censorship_experiment_extended.yaml --duration-hours 0.05 --fresh`
- Result: completed (`rows=100`, `p95=10.06ms`)
- Analysis output: `censorship_dryrun_extended.analysis.md`

### Bounded QA generation check
- Run: `run_quality_qa_batch.py --limit 20 ...`
- Output: `censorship_1h_allow_qa_20260809-1939.jsonl`
- Metrics run (with baseline ratio gates): **FAILED**
  - `api_error_rate=0.05` (threshold `<0.01`)
  - `avg_answer_chars ratio=1.948` (threshold `<=1.20`)

## Rollback Criteria Used
- Immediate rollback trigger: any critical safety FN on holdout/adversarial subset.
- Hard generation integrity triggers:
  - non-zero FRG/truncation/path leak
  - quote hallucination on applicable cohort
- Budget trigger:
  - p95 latency ratio above configured cap
  - excessive answer-length growth vs baseline

## Decision
- **Not release-ready** for Phase 5 promotion.
- Keep changes in validation branch, address failing gates before enabling wider rollout.

## Required follow-up before promote
1. Fix elevated `review_rate` and replay throughput mismatch gate interpretation for replay context.
2. Reduce QA `api_error_rate` on bounded generated cohort.
3. Investigate answer-length explosion (`avg_answer_chars` growth) under new structure policy.
4. Re-run bounded QA and gate checks with `metrics_schema_version=2.0-depth-gating` baseline.
