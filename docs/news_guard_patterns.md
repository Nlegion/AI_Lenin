"""NewsGuard input-gate pattern freeze (P0 / post-eval roadmap).

Source of truth for pattern strings remains [`config/news_guard.yaml`](../config/news_guard.yaml).
Stable snapshot for manual rollback: [`config/stable/news_guard.yaml`](../config/stable/news_guard.yaml).
Rollback: `python scripts/safety/rollback_gate_config.py restore` (primary=maintainer, backup=architect).

## Decision layers

| Layer | Decision | Notes |
|-------|----------|-------|
| combat co-occurrence (±10) + military phrases | deny | stems + military_co_tokens; metaphor blockers |
| military_topics (token-safe `сво`) | deny | not substring of `свои`/`свободн*` |
| hard_deny_keywords | deny | extremism |
| FIO + charge context | deny | toponym FP ignored unless charge markers |
| title/lead policy (≥2 markers or key+action-verb allowlist) | allow (full) | not single-word «Кремль» |
| body density≥1.0∧abs≥2 or unique≥3∧key | allow (full) | unique = marker **types** |
| primary sport/science/crime/disaster | **skip** (typed soft template) | policy-exception → full unless intra-domain negatives |
| primary social/labor/economy/geopolitics | allow (full) | social prompt extras |
| economy yellow carve-out | allow + `risk_tier=yellow` | economy markers && !combat && !RF ops; update markers via CR |
| quarantine_topics (`национальн` w/ excludes) | quarantine (or yellow allow if economy) | not «национальная компания» |
| allow_topics / unknown | allow / classify_on_unknown | live soft-pass unknown only |

Economy/policy marker lists: `input_gate.economy_policy_markers` + `allow_topics` in `news_guard.yaml`.
Quote grounding: [`docs/quote_grounding.md`](quote_grounding.md). Feature flags: `config/quality_postcheck.yaml`.
Red gold: `data/eval/red_gold_combat.jsonl` (≥50).

## Token-safe matching

| Pattern | Rule |
|---------|------|
| `спорт` | word-boundary; not inside `экспорт`/`транспорт` |
| `сво` | token/phrases (`в рамках сво`, `ход сво`, …); not `свои` |
| `национальн` | substring with exclude contexts (компания/проект/экономика) |

## Combat calibration

- Set: `tests/fixtures/quality/combat_calib_30.jsonl`
- Script: `python scripts/safety/calibrate_combat_gate.py`
- Target: F1 ≥ 0.90 for combat deny and indirect non-deny; expand to 50 if miss

## Conditional quote mode

- Quote-require if quote-span in top-K=3 chunks **and** lexical overlap ≥ 0.15
- `overlap = |lemmas(news) ∩ lemmas(chunk)| / |lemmas(news)|` (pymorphy3 when available)
- Else principles; social+empty R1: facts-first, no fabricated quotes
- Postcheck: strip quotes if answer has quotes but context has none

## Rate drift response (batch / release_pass)

| Drift | SLA (business hours) | Action |
|-------|----------------------|--------|
| 20–40% | ≤1 hour review | confirmed regression → manual rollback |
| >40% | ≤15 minutes | `rollback_gate_config.py restore` then postmortem |
| any | — | **no** auto/cron rollback |

Owners: primary=maintainer, backup=architect.

## Expansion checklist

1. Update yaml + this freeze table.
2. Run `pytest tests/test_news_guard_p0_regressions.py tests/test_news_guard.py -q`.
3. Run combat calib; snapshot stable on pass.
4. CHANGELOG metrics note.
"""
