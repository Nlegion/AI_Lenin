# Censorship Run Analysis

## Summary
- rows: 1440
- hard_block_rate: 48.19%
- review_rate: 13.40%
- skip_rate: 1.88%
- allow_rate: 36.53%
- reason_coverage: 100.00%
- p50_latency_ms: 4.22
- p95_latency_ms: 12.28

## Policy Core Slice
- n_policy_core: 359
- policy_core_share: 24.93%

## Decision Distribution
- allow: 526
- hard_block: 694
- review: 193
- skip: 27

## Top Categories
- NON_TOPICAL: 380
- None: 344
- WAR_OPERATIONAL: 305
- SPORT_BLOCKED: 285
- WAR: 40
- AIRPORT: 33
- DIPLOMACY: 16
- PROTESTS: 14
- PERSONAL_DATA: 13
- SANCTIONS: 7
- FIRE: 1
- DEATH: 1

## Top Reason Codes
- unknown_topic_low_signal_allow_forward: 379
- manual_war_operational_hard_block: 251
- unknown_topic: 169
- sport_blocked: 158
- manual_sport_hard_block: 127
- manual_war_hard_block: 40
- primary:social: 40
- context:military_rf_forces: 38
- manual_airport_hard_block: 33
- out_of_scope:crime: 21
- национальн: 17
- primary:labor_economy: 16
- primary:geopolitics: 15
- истор: 14
- abs:2: 14

## Live Decision Rates
- rows: 1200
- hard_block_rate: 49.58%
- review_rate: 10.33%
- skip_rate: 1.50%
- allow_rate: 38.58%
- reason_coverage: 100.00%

## Control Decision Rates (All Rows)
- rows: 240
- hard_block_rate: 41.25%
- review_rate: 28.75%
- skip_rate: 3.75%
- allow_rate: 26.25%
- reason_coverage: 100.00%

## Control Decision Rates (Unique news_id)
- rows: 240
- hard_block_rate: 41.25%
- review_rate: 28.75%
- skip_rate: 3.75%
- allow_rate: 26.25%
- reason_coverage: 100.00%

## L1 To Final Mismatches
- mismatch_rows: 386
- review->allow: 379
- allow->review: 6
- hard_block->review: 1

## Sources
- TASS: 1200
- CONTROL::rus_news_classifier: 86
- CONTROL::lenta_kaggle: 86
- CONTROL::ru_ethno_hate: 68

## Dataset Split
- live: 1200
- control: 240

## Compare Baseline
- compare_jsonl: p:\AI_Lenin\.cursor\artifacts\quality\censorship_90m_latest.jsonl

### Current Live (Unique N/A)
- rows: 1200
- hard_block_rate: 49.58%
- review_rate: 10.33%
- skip_rate: 1.50%
- allow_rate: 38.58%
- reason_coverage: 100.00%

### Compare Live (Unique N/A)
- rows: 1800
- hard_block_rate: 55.17%
- review_rate: 3.00%
- skip_rate: 2.28%
- allow_rate: 39.56%
- reason_coverage: 100.00%

### Current Control Unique
- rows: 240
- hard_block_rate: 41.25%
- review_rate: 28.75%
- skip_rate: 3.75%
- allow_rate: 26.25%
- reason_coverage: 100.00%

### Compare Control Unique
- rows: 20
- hard_block_rate: 30.00%
- review_rate: 30.00%
- skip_rate: 5.00%
- allow_rate: 35.00%
- reason_coverage: 100.00%

### Compare Sample Sizes
- current_control_rows_all: 240
- compare_control_rows_all: 360
- current_control_unique: 240
- compare_control_unique: 20
