# QA debris scrub hardening (live QA 20260813-2229)

## Plan
- Extend golden fixtures from QA-2229 debris cases
- Harden `answer_body_cleanup` (broken stance, md debris, prompt-task tail, bold labels)
- Strip visible `«[место]»` / `[обезличено]` in `final_public_scrub`
- Fix QA display formatter orphan `*`
- Add integrity residual codes + unit/replay tests

## Assumptions
- Soft integrity remains default
- Pipeline order unchanged: body cleanup → NewsGuard → final_public_scrub → display
- PII is not restored when markers are stripped

## Questions
- None

## Result
### Code
- `src/core/generation/answer_body_cleanup.py` — broader stance/`core_` scrub, terminal md debris, prompt-task multi-marker tail, section-boundary bold labels, integrity codes (`residual_stance`, `md_debris`, `prompt_task_echo`, `mesto_marker`)
- `src/core/generation/output_artifacts.py` — `final_public_scrub` strips mesto/obezlicheno + collapses empty quotes/double punct; `_PLACEHOLDER_MESTO_ALLOWED=False`
- `scripts/_quality_qa_txt.py` — normalize `**Label:**` before split; drop orphan `*+` chunks

### Fixtures
- `tests/fixtures/answer_postprocess/{core_lenin_broken_stance,md_hash_dash_tail,prompt_task_tail,bold_section_labels,mesto_placeholder}.{in,out}.txt`
- `tests/fixtures/answer_postprocess/qa2229_fixture_ids.json`

### Tests
- Extended `test_answer_body_cleanup.py`, `test_output_artifact_hardening.py`, `test_quality_qa_batch_io.py`
- Updated `test_trial50_hotfixes.py` (public text no longer keeps `«[место]»`)
- Added `tests/test_qa2229_debris_replay.py` (27 done answers → 0 debris hits)

### Verification
- `python -m alembic upgrade head`
- `pytest` targeted suite: **48 passed**
- `ruff check` clean; `ruff format` applied on touched files
