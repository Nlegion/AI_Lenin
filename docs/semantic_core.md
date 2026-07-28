# Semantic Core

Config-driven bridge from modern news surfaces to Lenin-register retrieval terms.

## Problem

Lenin did not write about neural networks, but he wrote about technical progress.
Filtering modern topics abandons the product goal (transfer logic, not quote matching).

## Flag interaction

| `semantic_core.enabled` | `dialectical_orchestration.enabled` | Effect |
|---|---|---|
| false | * | no-op |
| true | false | optional legacy enrich if `apply_to_legacy`; synthesis hint if routed |
| true | true | abstract slot queries when `apply_to_dialectical` and non-empty terms |

`semantic_core` never auto-enables dialectical orchestration.
When dialectical is OFF, hints without matching R1 context are limited; enable dialectical for full effect.

## Config

Source of truth: [`config/semantic_core.yaml`](../config/semantic_core.yaml).

Key knobs:
- `apply_to_dialectical` / `apply_to_legacy` (legacy default **false**)
- `include_axes_in_semantic_query` (default false)
- `include_title_anchor` (title **after** terms; terms never displaced)
- `max_query_chars`, `max_terms_per_topic`, `max_term_tokens`
- `empty_r1_fallback_to_legacy_slot_query`
- compound cliché gate: `cliche_warn_rate_max_ratio` × `cliche_warn_rate_min_delta_pp`
- `author_known_rate_min` for legacy A/B decisions

There is **no** vendor-specific `gpt_family` matcher. Triggers are domain phrases / charset-boundary short tokens.

## Routing

[`src/core/analysis/topic_router.py`](../src/core/analysis/topic_router.py):
- dominant topic by sum(weight), then max single-trigger weight, then YAML order
- multi-topic → structured log `semantic_core_multi_topic` with `run_id` + sha256 `title_hash`
- `hint_only` topics inject synthesis hint only (no retrieval terms; no legacy enrich)
- trigger `match` modes: `phrase` (multi-word `\s+`), `charset_boundary` (short tokens), `stem` (morphological endings)

## Query compose

[`src/core/analysis/semantic_query.py`](../src/core/analysis/semantic_query.py):
- terms joined with single spaces; truncate by whole term; spaces count toward budget
- `retrieval_terms` keep corpus orthography (yo-normalize is **routing-only**)

## Exhausted fallback

If abstract R1 is empty, retry legacy slot queries. If still empty:
- `semantic_fallback_exhausted=true`
- synthesis hint stripped
- `CONTEXT_UNAVAILABLE_MESSAGE`

## Lacuna hedge gate

Warn-only, non-mutating (same philosophy as cliché gate):
- writes `gate_warn_audit` with gate=`lacuna_hedge`
- metadata `lacuna_hedge_warn` + `matched_patterns`

## Retokenize after YAML length changes

When changing `retrieval_terms`, `max_terms_per_topic`, `max_term_tokens`, `max_query_chars`, or title-anchor settings:

```powershell
python scripts/calibrate_semantic_core_query.py
```

Compare token counts to `model_max_tokens - embedder_token_margin` and update `max_query_chars` if needed.
CI token smoke is a recommended post-MVP follow-up.

## hint_only vs drop

After Phase 0 corpus probe:
- keep topic as `hint_only: true` if `synthesis_hint` is analytically useful
- otherwise remove the topic from YAML
- record the decision in the Phase 0 artifact

## Legacy A/B

`apply_to_legacy` stays **false** when `author_known_rate < author_known_rate_min` and human scores are unavailable.
Primary Lenin metric when known-rate is sufficient: `lenin_share_known` via `is_lenin_author`.

## Evaluation

```powershell
python scripts/evaluate_semantic_core.py
```

Fixtures: `data/eval/semantic_core_fixtures.jsonl`.
