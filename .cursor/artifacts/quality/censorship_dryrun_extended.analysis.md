# Censorship Run Analysis

## Summary
- rows: 100
- hard_block_rate: 54.00%
- review_rate: 9.00%
- skip_rate: 2.00%
- allow_rate: 35.00%
- reason_coverage: 100.00%
- p50_latency_ms: 0.19
- p95_latency_ms: 10.06

## Policy Core Slice
- n_policy_core: 100
- policy_core_share: 100.00%

## Decision Distribution
- hard_block: 54
- allow: 35
- review: 9
- skip: 2

## Top Categories
- NON_TOPICAL: 29
- SPORT_BLOCKED: 23
- WAR_OPERATIONAL: 22
- None: 15
- WAR: 4
- AIRPORT: 3
- DEATH: 1
- SANCTIONS: 1
- PROTESTS: 1
- DIPLOMACY: 1

## Top Reason Codes
- unknown_topic_low_signal_allow_forward: 29
- override:unknown_topic_forward_trusted_source: 29
- manual_war_operational_hard_block: 19
- manual_sport_hard_block: 13
- sport_blocked: 10
- unknown_topic: 7
- manual_war_hard_block: 4
- manual_airport_hard_block: 3
- primary:labor_economy: 2
- out_of_scope:crime: 2
- context:military_rf_forces: 2
- primary:social: 2
- manual_death_hard_block: 1
- мобилизац: 1
- sanctions_allow_gate: 1

## Live Decision Rates
- rows: 100
- hard_block_rate: 54.00%
- review_rate: 9.00%
- skip_rate: 2.00%
- allow_rate: 35.00%
- reason_coverage: 100.00%

## Control Decision Rates (All Rows)
- rows: 0
- hard_block_rate: 0.00%
- review_rate: 0.00%
- skip_rate: 0.00%
- allow_rate: 0.00%
- reason_coverage: 0.00%

## Control Decision Rates (Unique news_id)
- rows: 0
- hard_block_rate: 0.00%
- review_rate: 0.00%
- skip_rate: 0.00%
- allow_rate: 0.00%
- reason_coverage: 0.00%

## L1 To Final Mismatches
- mismatch_rows: 30
- review->allow: 29
- allow->review: 1
- intentional_override_rows: 29
- unexpected_conflict_rows: 1

## Sources
- TASS: 100

## Dataset Split
- live: 100

## Compare Baseline
- compare_jsonl: p:\AI_Lenin\.cursor\artifacts\quality\censorship_1h_baseline.jsonl

### Current Live (Unique N/A)
- rows: 100
- hard_block_rate: 54.00%
- review_rate: 9.00%
- skip_rate: 2.00%
- allow_rate: 35.00%
- reason_coverage: 100.00%

### Compare Live (Unique N/A)
- rows: 1200
- hard_block_rate: 49.58%
- review_rate: 10.33%
- skip_rate: 1.50%
- allow_rate: 38.58%
- reason_coverage: 100.00%

### Current Control Unique
- rows: 0
- hard_block_rate: 0.00%
- review_rate: 0.00%
- skip_rate: 0.00%
- allow_rate: 0.00%
- reason_coverage: 0.00%

### Compare Control Unique
- rows: 240
- hard_block_rate: 41.25%
- review_rate: 28.75%
- skip_rate: 3.75%
- allow_rate: 26.25%
- reason_coverage: 100.00%

### Compare Sample Sizes
- current_control_rows_all: 0
- compare_control_rows_all: 240
- current_control_unique: 0
- compare_control_unique: 240
