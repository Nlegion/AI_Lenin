"""NewsGuard input-gate pattern freeze (P0.2.0).

Source of truth for pattern strings remains [`config/news_guard.yaml`](../config/news_guard.yaml).
This document freezes **roles** and **expansion policy**. Do not add broad roots
(`социальн`, `школ`, `медицин`) without must_answer FP check + CHANGELOG.

## Decision layers

| Layer | Decision | Must-refuse target? |
|-------|----------|---------------------|
| `military_topics` | deny | Yes |
| `hard_deny_keywords` | deny | Yes |
| `quarantine_topics` / `quarantine_keywords` | quarantine (≡ deny on hot path + QA batch) | Yes |
| `hard_deny_topics` (sport/show/…) | deny | **No** (content-type filter) |
| `allow_topics` | allow when matched | must_answer fixtures should hit allow |
| `classify_on_unknown_as: quarantine` | quarantine | Risk for unbaited news; eval fixtures must be allow- or deny-covered |

## Hot-path / QA batch policy

- `deny` and `quarantine` → **no LLM**; `blocked=true`; `skipped_llm=true`
- `skipped_llm_reason`: `pre_deny` | `pre_quarantine`
- Refusal text = `refusal_message` from yaml (not model-parroting)

## Known false-positive risks (do not expand casually)

| Pattern | Risk |
|---------|------|
| `сво` in `military_topics` | substring of «свободн*», «своё» |
| `национальн` in `quarantine_topics` | «национальная экономика/проект» |
| `военн` in `high_risk_topics` | broad; not hard deny alone |

## Expansion checklist

1. Update yaml + this freeze table.
2. Run must_answer_12: zero new pre-gate FP.
3. Add CHANGELOG metrics note.
"""
