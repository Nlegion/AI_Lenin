# postprocess_clean baseline (2026-08-14)

Audit map before extracting a unified two-phase postprocess module.

## Standard path

`NewsProcessor` → `generate_and_persist_analysis` → `LeninAnalyzer.generate_analysis`
→ `AnalysisGenerationPipeline._generate_with_context`:

1. `finalize_generated_text`
2. `apply_quality_post_generate` (quote → loop → `apply_artifact_pass`)
   - `apply_artifact_pass`: encoding/citation/service strip → `cleanup_answer_body` → early `final_public_scrub`
3. `clamp_answer_length`
4. `NewsGuard.mark_unverified_facts`
5. `apply_post_generate_gates` (`NewsGuard.guard_output`)
6. optional yellow warning
7. terminal `final_public_scrub`

## Reasoning-publish path

`apply_post_qc_for_reasoning` (`skip_structure_enforce=True`) then the same
guard / yellow / terminal scrub. `suppress` timeout may skip post-guard and
return `CONTEXT_UNAVAILABLE_MESSAGE`.

## Late mutations after PipelineResult

1. `LeninAnalyzer.clean_analysis` — TextCleaner + rhetorical regex + `.`-split
   (metadata still describes pre-`clean_analysis` text).
2. `news_item_pipeline.generate_and_persist_analysis` — second `guard_output`
   with **no** following public scrub (can re-insert `«[место]»`).
3. `NewsProcessor.publish_cycle` — third `guard_output` without public scrub.

## Dual-rule overlap (not dual-phase)

`cleanup_answer_body` and `final_public_scrub` share encoding / hole / whitespace
patterns. Early `final_public_scrub` inside `apply_artifact_pass` plus terminal
`final_public_scrub` after Guard is **intentional** for NewsGuard PII markers.

## Publish gate

`is_publishable_analysis` uses `structure_error`, `postprocess_hard_fail`,
dialectical hold/suppress, placeholders. NewsGuard `blocked` is separate.
