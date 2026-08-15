# Censorship Auto-Rollback Runbook

## Automated rollback trigger
Use baseline-relative gate signals from:
- current run metrics (`*.metrics.json`)
- latency baseline (`censorship_latency_baseline.json`)
- thresholds in `config/release_gates.yaml`

When degraded, rollback action is:
- `shadow_mode: true`
- `enforce_mode: old`

## Automated command
```powershell
python scripts/safety/auto_rollback_censorship_shadow.py `
  --metrics-json .cursor/artifacts/quality/censorship_dryrun_short.metrics.json `
  --baseline-json .cursor/artifacts/quality/censorship_latency_baseline.json `
  --release-gates config/release_gates.yaml `
  --safety-config config/safety_gate_config.yaml
```

## Manual fallback
1. Set in `config/safety_gate_config.yaml`:
   - `shadow_mode: true`
   - `enforce_mode: old`
2. Re-run replay + gates:
   - `python scripts/safety/replay_censor_from_artifact.py ...`
   - `python scripts/safety/check_censorship_gates.py ...`
3. If quality remains degraded, restore stable snapshot:
   - `python scripts/safety/rollback_gate_config.py restore`

## Verification checklist
- Gate check returns `OK gates passed`.
- No red leakage increase in sampled review.
- Throughput ratio is above threshold.
- Review queue does not breach SLA.

