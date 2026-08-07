# Trial50 hotfix notes

## Feature flags

Configured under `quality_postcheck.trial50_hotfixes` in [`config/quality_postcheck.yaml`](../config/quality_postcheck.yaml).

- Masters: `safety_hotfixes_enabled`, `generation_hotfixes_enabled`
- Safety children: `drone_deny_enabled`, `combat_adjacent_softpass_block`, `sport_token_bound_enabled`, `fio_carveout_enabled`
- Generation children: `loop_strip_enabled`, `encoding_scrubber_enabled`, `disclaimer_footer_enabled`

Dependency: `combat_adjacent_softpass_block` covers the live soft-pass path for drone/combat-adjacent unknowns (shared helpers in `src/core/safety/drone_combat_guard.py`).

## Soft-pass outcome

When soft-pass is denied: quarantine / no-publish (`blocked` row, LLM skipped). Never generate.

## Encoding policy

Detect-first for `СЃ`, standalone `Рё`, Latin islands. Prefer fallback over blind `СЃ→США` / `Рё→его`.

## Next iteration (documented, out of scope)

- Hybrid intel / geopolitical ops (e.g. detentions abroad framed as info-war) may need a dedicated category beyond crime carve-out.
- Anti-cliché regen-on-threshold remains deferred; lexicon stays `warn_only`.
- Structural «Ленин подчеркивал…» depth is not solved by this hotfix.

## Shadow / verify

```powershell
python scripts/shadow_trial50_safety.py --input .cursor/artifacts/quality/live_news_qa_trial50_20260805-2119.jsonl
pytest tests/test_trial50_hotfixes.py tests/test_publisher_disclaimer.py -q
python scripts/calibrate_combat_gate.py
```

Rollback: disable offending child flag or master if safe-gold false-deny rises above baseline+2pp, `disclaimer_missing>0`, or fallback share spikes.
