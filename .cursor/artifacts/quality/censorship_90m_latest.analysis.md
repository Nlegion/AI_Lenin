# Censorship Run Analysis

## Summary
- rows: 2160
- hard_block_rate: 50.97%
- review_rate: 7.50%
- skip_rate: 2.73%
- allow_rate: 38.80%
- reason_coverage: 100.00%
- p50_latency_ms: 2.03
- p95_latency_ms: 8.41

## Policy Core Slice
- n_policy_core: 161
- policy_core_share: 7.45%

## Decision Distribution
- allow: 838
- hard_block: 1101
- review: 162
- skip: 59

## Top Categories
- WAR_OPERATIONAL: 668
- NON_TOPICAL: 567
- None: 523
- SPORT_BLOCKED: 244
- WAR: 85
- AIRPORT: 28
- DEATH: 18
- PERSONAL_DATA: 16
- FIRE: 11

## Top Reason Codes
- manual_war_operational_hard_block: 632
- unknown_topic_low_signal_allow_forward: 567
- unknown_topic: 144
- manual_sport_hard_block: 125
- sport_blocked: 119
- manual_war_hard_block: 85
- primary:social: 76
- primary:labor_economy: 51
- abs:3: 36
- manual_airport_hard_block: 28
- out_of_scope:crime: 27
- manual_death_hard_block: 18
- title_lead_policy:президент,путин: 18
- title_lead_policy:депутат,парламент,чиновник: 18
- истор: 18

## Sources
- TASS: 1800
- CONTROL::rus_news_classifier: 180
- CONTROL::ru_ethno_hate: 90
- CONTROL::lenta_kaggle: 90

## Dataset Split
- live: 1800
- control: 360
