# SafetyGate Operations Walkthrough

## Enforcement modes

| Mode | Behavior |
|------|----------|
| `enabled=false` | Legacy `NewsGuard` only |
| `shadow_mode=true` | Dual-run; **enforce old** decisions; log mismatches |
| `enforce_mode=new` + `shadow_mode=false` | Enforce SafetyGate decisions |

## Rollback switches

1. Generation template pressure: `config/quality_postcheck.yaml`
   - `loop_fix_enabled`
   - `yellow_output_filter_enabled`
   - `quote_postcheck_enforce_mode` / `artifact_enforce_mode`
2. Safety: `config/safety_gate_config.yaml` flags + `trial50_hotfixes` children
3. Stable snapshot: `config/stable/news_guard.yaml` via `scripts/safety/rollback_gate_config.py`

## Metrics (must-have)

- `gate_allow_share`, `gate_deny_share`, `gate_skip_share`, `gate_yellow_share`
- `template_fallback_share`, `static_safe_template_share`
- `quote_repair_applied_rate`, `repair_success_rate`
- `safety_gate_latency_ms` p50/p95
- Alert levels via `src/core/safety/safety_gate_metrics.alert_levels`

## Dashboard slices

Break down by risk bucket (`red`/`yellow`/`green`) and topic bucket:

- gate decisions
- template fallback share
- output length / depth proxies
- parity mismatch by reason code

## Canary promote criteria

- red leakage = 0%
- refusal delta within ~1pp (or agreed CI band)
- depth proxies improved; cleanliness not regressed
- sample size >= agreed window (e.g. 500 items)
