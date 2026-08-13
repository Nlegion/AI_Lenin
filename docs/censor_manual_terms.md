# Manual pre-RAG censor terms

Source of truth for keyword hard-blocks used by `PreRagCensor`.

## Layout

- `config/censor_terms/index.yaml` — category order, `enabled`, `decision`, `reason_code`, file name
- `config/censor_terms/<category>.yaml` — `terms:` list only
- `config/censor_terms/overrides.yaml` — `force_include` / `force_exclude` without code changes
- Draft lists: `models/words.txt` (not loaded at runtime)

## Matching

- Text is normalized once; terms are stored casefolded
- Match is substring: `term in text_lower` (same as legacy manual lists)
- Evaluation order: ethno-hate and air-alert specials in code, then `index.yaml` order
- First hit wins

## Sport exception

`SPORT_BLOCKED` is the only category gated by runtime config (`sport_block_enabled` in `CensorRuntimeConfig` / `safety_gate_config.yaml`). All other categories use `enabled` in `index.yaml` only.

## Adding or removing terms

1. Prefer editing the category YAML under `config/censor_terms/`
2. Or use overrides:

```yaml
force_include:
  - category: WAR_OPERATIONAL
    term: new-term
force_exclude:
  - category: FOOD
    term: ambiguous-term
```

3. Keep whole-term precision: do not add short ambiguous stems from `src/core/safety/manual_terms_policy.py` (`AMBIGUITY_BANLIST`)
4. Rebuild from draft only when intentionally re-scrubbing:

```powershell
.venv\Scripts\python.exe scripts/build_censor_terms_from_draft.py
```

Review the scrub artifact under `.cursor/artifacts/*-censor-terms-scrub.md`.

## FP review checklist

- Economic/politics news should not trip lifestyle categories via stems like `игра`, `пост`, `золото`
- War frontline toponyms belong in `WAR_OPERATIONAL`; leisure toponyms (`сочи`, bare `крым`) stay out
- No `TRANSPORT` category; RZD/GIBDD terms are dropped from airport
- Crime stems are not part of `SHOWBIZ`

## Hot reload

When `hot_reload_enabled` is true, the censor polls and reloads terms with last-good retention on parse errors. `manual_terms_hash` is folded into `config_version_hash`.
