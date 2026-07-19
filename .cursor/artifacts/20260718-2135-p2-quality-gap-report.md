# P2 Quality Gap Closure Report

## Inputs
- Baseline sandbox: `.cursor/artifacts/sandbox/retrieval_sandbox_summary.md`
- Tuned sandbox: `.cursor/artifacts/sandbox/retrieval_sandbox_summary_tuned.md`
- Baseline quality: `.cursor/artifacts/evaluation/rag_quality_summary.md`
- Tuned quality: `.cursor/artifacts/evaluation/rag_quality_summary_tuned.md`
- Fundamental embedding check: `.cursor/artifacts/embeddings/embedding_selection_fundamental.md`
- Foundation audit: `.cursor/artifacts/evaluation/retrieval_foundations_audit.md`

## Tuning Results

| Metric | Baseline | Tuned | Target |
|---|---:|---:|---:|
| Recall@5 | 0.0250 | 0.0417 | 0.85 |
| MRR@10 | 0.0179 | 0.0212 | n/a |
| nDCG@10 | 0.0235 | 0.0299 | n/a |
| Core self ratio | 1.0000 | 0.0833 | 0.70 |
| Ideology consistency | 0.0000 | 0.0000 | 0.70 |
| Empty context rate | 0.0000 | 0.0000 | <=0.10 |

### Observations
- Tuning weights/top_k/rrf improved retrieval relevance modestly (`Recall@5` +0.0167 absolute).
- Improvement came with a strong drop in `core_self_ratio`, showing unstable source-balance behavior.
- `ideology_consistency` remains `0.0`, indicating gap is structural, not only fusion-hyperparameter related.

## Fundamental Remediation (P2b)
1. **Embedding reassessment**
   - Tested `all-MiniLM-L6-v2` vs `BAAI/bge-m3`.
   - `bge-m3` failed on available RAM allocation during CPU load (`alloc_cpu` OOM).
   - Winner remains `all-MiniLM-L6-v2`, but recall remains far below target.
2. **Ontology tag audit**
   - Registry and ontology tags are aligned (`148/148`, no missing sources).
   - Stance distribution preserved (document-level and chunk-level).
3. **Chunk integrity audit**
   - `chunking_summary`: boundary ratio `0.0000`, compliance `0.9990`.
   - No evidence of large-scale boundary corruption.
4. **HyDE + query rewriting**
   - Maintained enabled in tuned retrieval pipeline.
   - HyDE was beneficial in baseline sandbox but not dominant after tuned weighting.

## Root-Cause Hypothesis
- Eval dataset positives and current embedding space remain weakly aligned on full corpus scale.
- Hardware-bound model choice (CPU-only, local RAM constraints) prevents high-capacity embedding baseline.
- Ideology consistency metric likely needs stronger semantic classifier/judge pipeline rather than retrieval-only tuning.

## Decision
- Acceptance thresholds are **not met** after tuning + fundamental checks.
- Continue with controlled progression to migration/operations only under **research mode** (not production cutover).

## Proposed Interim Targets (for next loop, pending user approval)
- Recall@5: `>= 0.05`
- Core self ratio: `>= 0.50`
- Ideology consistency: `>= 0.20`

These are interim engineering gates until:
- memory-safe higher-capacity embedding adaptation is available, and
- ideology consistency evaluator is improved beyond current proxy behavior.
