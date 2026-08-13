# 20260813 — Answer body cleanup v1

## Plan
Implemented pre-guard answer body cleanup per revised postprocess plan.

## Result
- Added `src/core/generation/answer_body_cleanup.py` (stance/instruction scrub, header normalize, exact trailing triad truncate, soft integrity).
- Wired into `apply_artifact_pass` before token-only `final_public_scrub`.
- Config: `answer_body_cleanup_enabled`, `integrity_check_enabled`, `integrity_enforce_mode` in `quality_postcheck.yaml` + pydantic model.
- `publishability.py` honors `postprocess_hard_fail`.
- `pipeline.py` resyncs `dialectical_outcome` / reason codes after post-QC hold (and hard-fail).
- Golden fixtures under `tests/fixtures/answer_postprocess/`.
- Tests: `test_answer_body_cleanup.py`, extended hardening + publishability.

## Verification
```
python -m pytest tests/test_answer_body_cleanup.py tests/test_output_artifact_hardening.py tests/test_depth_publishability.py tests/test_quality_hardening_helpers.py tests/test_quality_soft_repair.py tests/test_quality_remarks_regressions.py -q
```
39 passed.

## Deferred (v2)
- `evaluate_answer_postprocess.py`, release_gates postprocess section, strict integrity default, yellow/disclaimer reorder.
