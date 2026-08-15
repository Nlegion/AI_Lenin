# postprocess_clean cutover (2026-08-14)

## Mode

`config/quality_postcheck.yaml` → `postprocess_clean_mode: live`

- `live` — production writer is `run_postprocess` / `apply_terminal_public_scrub`
- `shadow` — legacy writer on the live string; clone compared to the new module; jsonl at `.cursor/artifacts/quality/postprocess_clean_shadow.jsonl`
- `off` — legacy two-call writer only (rollback)

Rollback: set `postprocess_clean_mode: off` and restart. Do not run two writers on the same live string.

## Writer inventory

| Call site | Phase |
|-----------|--------|
| `apply_artifact_pass` | `pre_guard` via `apply_pre_guard_for_artifact` |
| `pipeline.py` (standard + reasoning-publish) | `post_guard` via `apply_terminal_public_scrub` |
| `news_item_pipeline.generate_and_persist_analysis` | `scrub_after_output_guard` after persist `guard_output` |
| `processor.publish_cycle` | `scrub_after_output_guard` after publish `guard_output` |

## Cleanup remaining (later)

Legacy transformers stay in `answer_body_cleanup.py` / `output_artifacts.py` as the rule source. Public tests still import `cleanup_answer_body` and `final_public_scrub`. Parser-first rewrite is out of v1.
