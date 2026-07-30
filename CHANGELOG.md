# Changelog

## Template

```markdown
## YYYY-MM-DD — <phase id> — <author>
### Changed
- …
### Metrics
| metric | before | after |
|--------|--------|-------|
| refusal_phrase_rate | … | … |
### Artifacts
- `.cursor/artifacts/quality/<batch>.jsonl`
### Notes
- …
```

## 2026-07-29 — qa-quality-hardening-p0 — agent

### Changed
- Removed TextCleaner `Рё→ФРГ` map (possessive `её` corruption).
- Rewrote GigaChat system prompt: no parrot-able safety refusal; news-anchor instruction.
- QA batch pre-LLM gate: `deny`/`quarantine` → `skipped_llm` + `skipped_llm_reason`.
- Chunk-first token budget shrink; silent context trim; `max_tokens` 300→512.
- Consecutive-sentence dedupe; news groundedness warn gate.
- Enabled `dialectical_orchestration` and `semantic_core` for eval.
- Added must_answer_12 / must_refuse fixtures, metrics script, gate pattern freeze doc.

### Metrics
| metric | before (batch 2046) | after (batch 0649 / must_answer_12) |
|--------|---------------------|--------------------------------------|
| refusal_phrase_rate | ~0.88 | **0.00** |
| frg_artifact_rate | ~0.28 | **0.00** |
| truncated_marker_rate | >0 | **0.00** |
| must_refuse_block_rate | n/a | **1.00** (pre-LLM) |
| news_groundedness_rate | n/a | **1.00** (full 50) |

### Artifacts
- `tests/fixtures/quality/must_answer_12.jsonl`
- `tests/fixtures/quality/must_refuse.jsonl`
- `.cursor/artifacts/quality/must_answer_12_20260729-0640.*`
- `.cursor/artifacts/quality/must_refuse_20260729-0640.*`
- `.cursor/artifacts/quality/quality_qa_batch_20260729-0649.*` (post-hardening full 50)
- `.cursor/artifacts/quality/quality_qa_batch_20260728-2046.*` (baseline)

### Notes
- Batch 2046 is **not** baseline for R1/semantic metrics (flags were OFF).
- Post-hardening full 50: all `orchestration_mode=legacy_fallback` (empty abstract R1 → legacy RAG); semantic dominant on 4/50. Follow-up: improve R1 retrieval hit-rate under semantic_core.
- See `docs/news_guard_patterns.md` for pattern freeze / expansion policy.
