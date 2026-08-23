# Changelog

## 2026-08-23 — tass-rss-and-r13-traces — agent
### Changed
- TASS RSS: browser User-Agent / Accept headers; HTTP ≥400 and bozo-empty feeds return no items (`tass_rss_http_error` / `tass_rss_parse_failed`).
- EvidenceBrief slot traces (`r1_items` / `r2_items` / `r3_items`, `SLOT_TRACE_TEXT_CAP=800`) in pipeline metadata; QA runtime copies them into JSONL.
- CLI: `scripts/quality/run_r13_example_trace.py` dumps news + R1/R2/R3 chunks + LLM answer (empty R3 rendered as `(пусто)`).
- Docs: architecture PNGs, DeepSeek prompt split vs llama, VPS TASS/StormWall note, R1–R3 status marked implemented.
### Metrics
| metric | note |
|--------|------|
| StormWall 403 | feedparser default UA blocked; Mozilla/Chrome UA required |
| R3 empty | valid input; do not fabricate opposition |
### Artifacts
- `docs/architecture_offline.png`, `docs/architecture_online.png`
- `tests/test_tass_rss_headers.py`, `tests/test_r13_example_trace.py`
### Notes
- Do not set global `HTTP(S)_PROXY` on VPS (TASS/DeepSeek stay direct; SOCKS is Telegram-only).

## 2026-08-16 — vps-deepseek-publication — agent
### Changed
- LLM transport moved to `src/core/llm/`; `LLM_PROVIDER=deepseek` for VPS (no GGUF / llama-server in the container).
- DeepSeek-only prompt builders, R1 quote allowlist, one regenerate; Telegram triad parser accepts flattened `Факт`/`Механизм`/`Вывод`.
- Docker RAG replica, Telegram SOCKS egress (`TELEGRAM_PROXY_*`), PreRagCensor sport/lifestyle blocks, quieter admin telemetry.
- Legal pack already on this branch: MIT + research disclaimer wording.
### Metrics
| metric | note |
|--------|------|
| VPS generation | DeepSeek API; retrieval/gates unchanged |
| llama prompt | left intact; DeepSeek does not reuse triad/quote rules |
### Artifacts
- `docs/docker.md`, `docs/llm_client.md`, `.env.example`
### Notes
- Workstation default remains local GigaChat3 via llama-server.

## 2026-08-15 — legal-docs-attribution — agent
### Changed
- Added root legal pack: `LICENSE` (MIT, repo-authored code/docs only), bilingual `DISCLAIMER.md`, `THIRD_PARTY_LICENSES.md`, `CONTRIBUTING.md`.
- README: Legal notice, accurate PSS/data paths, DeepSeek provider table; AGENTS Tech Stack + legal SoT link; docs index; NOTICE audit script path.
- Unified Telegram `AI_DISCLAIMER` with `quality_postcheck.short_disclaimer` («исследовательских целях»).
### Metrics
| metric | note |
|--------|------|
| publisher vs NewsGuard footer | same educational wording SoT |
### Artifacts
- `LICENSE`, `DISCLAIMER.md`, `THIRD_PARTY_LICENSES.md`, `CONTRIBUTING.md`
- `tests/test_publisher_disclaimer.py`
### Notes
- Corpus digitization URL still unrecorded; whole PSS files not claimed PD.

## 2026-08-15 — triad-flatten-cleanup — agent
### Changed
- Restore triad section breaks after consecutive-sentence join in `finalize_generated_text`.
- Body cleanup: inline `Факт`/`Механизм`/`Вывод` spacing, same-line disclaimer split, empty `--- [empty] ---` scaffolds, trailing markdown before triad cut.
### Metrics
| metric | note |
|--------|------|
| triad labels after flatten | re-broken before quality_hooks / cleanup regexes |
### Artifacts
- `tests/test_answer_body_cleanup.py`
- `tests/test_quality_hardening_helpers.py`
### Notes
- `postprocess_clean_mode` remains `live`. README / AGENTS / docs index / CLI paths aligned to current runtime.

## 2026-08-14 — postprocess-clean-v1 — agent
### Changed
- Unified `postprocess_clean` contract (`pre_guard` / `post_guard`) as the live writer for body+public scrub.
- Terminal public scrub after persist/publish `NewsGuard.guard_output`; `LeninAnalyzer.clean_analysis` no longer re-mutates pipeline text.
- Config: `quality_postcheck.postprocess_clean_mode` (`live` default; `shadow` / `off` rollback).
### Metrics
| metric | note |
|--------|------|
| dual inner+outer public scrub | same rules, two phases; post_guard remains after Guard |
| late mutation after terminal scrub | removed (`clean_analysis` identity) |
### Artifacts
- `.cursor/artifacts/20260814-2110-postprocess-clean-baseline.md`
- `.cursor/artifacts/20260814-2130-postprocess-clean-cutover.md`
### Notes
- Quote/loop/NewsGuard stay outside the module. Soft integrity default unchanged.

## Template

```markdown
## YYYY-MM-DD — <phase id> — <author>
### Changed
- …
### Metrics
| metric | before | after |
|--------|--------|-------|
| refusal_phrase_rate | … | … |
### Artifacts
- `.cursor/artifacts/quality/<batch>.jsonl`
### Notes
- …
```

## 2026-07-30 — post-eval-quality-roadmap — agent

### Changed
- Token-safe NewsGuard matching (`спорт`/`сво`/`национальн` excludes); FIO toponym FP + charge keep.
- Output FIO redact **without** `IGNORECASE`; combat ±10 co-occurrence + metaphor blockers.
- Primary-topic routing: title/lead policy, body density/distinct, out-of-scope **skip**, social full.
- Conditional quote mode (top-K + lexical overlap ≥0.15); social prompt extras; quote postcheck strip.
- Pipeline metadata: `quote_mode`, `top_chunks` (id/score/sha256/text), routing flags.
- `config/stable/` snapshot + `scripts/rollback_gate_config.py`; combat calib script/fixtures.
- Batch metrics helpers + graded drift SLA docs (owners maintainer/architect).

### Metrics
| metric | target / note |
|--------|----------------|
| combat_f1 / indirect_f1 | ≥0.90 on combat_calib_30 (`scripts/calibrate_combat_gate.py`) |
| export/Audi/tennis FP | must_answer / skip — see `test_news_guard_p0_regressions` |
| redact_artifact_rate | case-sensitive FIO sub; warn code `redact_artifact_present` |

### Artifacts
- `tests/fixtures/quality/combat_calib_30.jsonl`
- `config/stable/news_guard.yaml`
- `.cursor/artifacts/safety/combat_calib_summary.json` (after calib run)

### Notes
- Live soft-pass unknown unchanged; skip → `out_of_scope_skip`; combat never soft-passed.
- Weekly must_* refresh: owner maintainer; pytest gate suite before merge.

## 2026-07-29 — qa-quality-hardening-p0 — agent

### Changed
- Removed TextCleaner `Рё→ФРГ` map (possessive `её` corruption).
- Rewrote GigaChat system prompt: no parrot-able safety refusal; news-anchor instruction.
- QA batch pre-LLM gate: `deny`/`quarantine` → `skipped_llm` + `skipped_llm_reason`.
- Chunk-first token budget shrink; silent context trim; `max_tokens` 300→512.
- Consecutive-sentence dedupe; news groundedness warn gate.
- Enabled `dialectical_orchestration` and `semantic_core` for eval.
- Added must_answer_12 / must_refuse fixtures, metrics script, gate pattern freeze doc.

### Metrics
| metric | before (batch 2046) | after (batch 0649 / must_answer_12) |
|--------|---------------------|--------------------------------------|
| refusal_phrase_rate | ~0.88 | **0.00** |
| frg_artifact_rate | ~0.28 | **0.00** |
| truncated_marker_rate | >0 | **0.00** |
| must_refuse_block_rate | n/a | **1.00** (pre-LLM) |
| news_groundedness_rate | n/a | **1.00** (full 50) |

### Artifacts
- `tests/fixtures/quality/must_answer_12.jsonl`
- `tests/fixtures/quality/must_refuse.jsonl`
- `.cursor/artifacts/quality/must_answer_12_20260729-0640.*`
- `.cursor/artifacts/quality/must_refuse_20260729-0640.*`
- `.cursor/artifacts/quality/quality_qa_batch_20260729-0649.*` (post-hardening full 50)
- `.cursor/artifacts/quality/quality_qa_batch_20260728-2046.*` (baseline)

### Notes
- Batch 2046 is **not** baseline for R1/semantic metrics (flags were OFF).
- Post-hardening full 50: all `orchestration_mode=legacy_fallback` (empty abstract R1 → legacy RAG); semantic dominant on 4/50. Follow-up: improve R1 retrieval hit-rate under semantic_core.
- See `docs/news_guard_patterns.md` for pattern freeze / expansion policy.
