# Generation Backend Comparison

- Generated at (UTC): `2026-07-19T05:58:38Z`
- Decision: `KEEP_BASE_STRONG_DEFAULT: safety_compliance acceptable relative to fine_tuned.`

## Aggregates

| Backend | Avg latency ms | Style hits | Hallucination rate | Safety block rate | Prohibited rate | Deny control pass |
|---|---:|---:|---:|---:|---:|---:|
| `base_strong` | 20981.33 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| `fine_tuned` | 20803.67 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

## Legal Residual Risks
- GigaChat3 may hallucinate facts; publication risk includes false factual claims.
- Residual prohibited-content risk remains despite NewsGate/NewsGuard.
- base_strong is less corpus-bound than fine_tuned; content responsibility is higher.
- Public publishing requires disclaimer + owner identification checklist and legal review.

JSON artifact: `P:\AI_Lenin\.cursor\artifacts\generation\backend_comparison.json`
