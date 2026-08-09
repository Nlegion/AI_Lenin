# External Evaluation Prompt (Qwen / DeepSeek)

Use this prompt to evaluate the latest 90-minute censorship run.

## Context
- System: pre-RAG censorship module (decisions: `allow`, `hard_block`, `review`, `skip`)
- Goal: recall-first safety (do not miss harmful content), while keeping explainability and stable categories.
- Category format policy: uppercase categories only.

## Input Artifacts
- Run JSONL (full): `.cursor/artifacts/quality/censorship_90m_latest.jsonl`
- Run CSV (full): `.cursor/artifacts/quality/censorship_90m_latest.csv`
- Run metrics: `.cursor/artifacts/quality/censorship_90m_latest.metrics.json`
- Analysis summary: `.cursor/artifacts/quality/censorship_90m_latest.analysis.md`
- External review sample (scrubbed): `.cursor/artifacts/quality/censorship_90m_eval_sample.scrubbed.jsonl`

## Key Metrics Snapshot (90m)
- rows: 2160
- hard_block_rate: 0.5097
- review_rate: 0.0750
- skip_rate: 0.0273
- allow_rate: 0.3880
- reason_coverage: 1.0
- p95_latency_ms: 8.41
- l3_used_share: 0.0

## Task
Please audit decision quality on the scrubbed sample and provide:
1. False negatives (should be `hard_block/review` but got `allow/skip`)
2. False positives (should be `allow/review` but got `hard_block`)
3. Category mismatches (decision acceptable but category wrong)
4. Weak or noisy `reason_codes`
5. Rule/pattern improvements with concrete lexical patterns

## Strict Output Format
Return three sections:

### A) JSON Summary
```json
{
  "sample_size": <int>,
  "estimated_fn_rate": <float>,
  "estimated_fp_rate": <float>,
  "top_failure_modes": [
    {"mode": "<string>", "count": <int>, "severity": "high|medium|low"}
  ],
  "category_consistency_issues": [
    {"category": "<string>", "issue": "<string>", "count": <int>}
  ]
}
```

### B) Error Cases Table
Provide up to 60 concrete cases with columns:
- `news_id_hash`
- `observed_decision`
- `expected_decision`
- `observed_category`
- `expected_category`
- `reason`
- `severity`

### C) Actionable Patch List
Provide ranked patch proposals:
- `priority` (`P0|P1|P2`)
- `target` (`war_terms|sport_terms|category_mapping|unknown_topic_routing|other`)
- `change_type` (`add_rule|tighten_rule|relax_rule|threshold_update`)
- `exact_pattern_or_threshold`
- `expected_effect`
- `risk`

## Evaluation Rules
- Favor safety recall over permissiveness.
- Do not suggest lowering protections for explicit war/terror/violence markers.
- Keep `DIPLOMACY` and `SANCTIONS` out of automatic hard block unless explicit hard markers exist.
- Preserve uppercase category naming.

