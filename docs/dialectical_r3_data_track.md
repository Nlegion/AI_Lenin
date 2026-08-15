# R3 corpus / Principle seed track

Goal: reduce systematic `r3_absent` and improve dialectic depth.

## Checklist

1. Audit `stance_type=influence_critical` coverage via `scripts/retrieval/audit_retrieval_foundations.py`.
2. Remap critical authors in `config/source_registry_rules.yaml`.
3. Re-ingest / re-index affected chunks.
4. Keep seed examples in `src/core/dialectics/fixtures/principle_seeds.json` (positive/negative).
5. Gate full dialectic eligibility on real R3 presence until coverage improves.

Until R3 is dense, engine honestly emits `r3_absent` and may take simplified path — do **not** invent opposition.
