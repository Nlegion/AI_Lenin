# Crisis hardening remaining — implementation note

Date: 2026-07-28

## Delivered

- H1 anti-cliche gate (`config/anti_cliche.yaml`, `cliche_gate.py`) wired in pipeline; warn-only; fail-open; always-call
- M2 `anachronism_gate` + `config/anachronism.yaml` + prompt rule
- M1 unified `config/release_gates.yaml`; `release_pass` CLI; eval scripts under `.cursor/artifacts/evaluation/`
- M3 `docs/human_eval_checklist.md` + `scripts/collect_anti_cliche_label_batch.py`
- H3 README / docs index / AGENTS updates
- Warn audit: `.cursor/artifacts/safety/gate_warn_audit.jsonl`

## Verification

```
pytest tests/test_cliche_gate.py tests/test_pipeline_cliche_gate.py tests/test_anachronism_gate.py tests/test_release_pass.py tests/test_release_gates.py tests/test_dialectical_evidence_brief.py tests/test_dialectical_pipeline.py tests/test_retrieve_by_stance_filter.py -q
# 43 passed
python scripts/evaluate_anti_cliche.py
```

Block mode remains off pending H1-d bar.
