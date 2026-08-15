# Censorship Legacy Decommission Plan

## Scope
- `src/core/safety/news_guard.py`
- `src/core/safety/safety_gate.py` legacy enforcement paths
- `src/core/safety/topic_routing.py` legacy routing overrides

## Decommission Phases
1. **Shadow-only window (2 weeks)**  
   Keep legacy logic for parity logs only, enforce pre-RAG censor decisions.
2. **Freeze and verify**  
   No new policy changes in legacy modules. Verify zero functional dependency in runtime path.
3. **Bypass legacy at runtime**  
   Route all pre-generation decisions through `PreRagCensor` only.
4. **Remove dead paths**  
   Delete unreachable legacy pre-input branches, keep output guard where explicitly needed.
5. **Post-removal soak**  
   One full canary window with rollback readiness.

## Exit Criteria
- `shadow_agreement_min` satisfied for canary window.
- No increase in red leakage.
- Latency/throughput gates pass relative to baseline.
- Review queue SLA remains within release thresholds.

## Rollback
- Immediate switch to `shadow_mode=true`, `enforce_mode=old` in `config/safety_gate_config.yaml`.
- Run `scripts/safety/rollback_gate_config.py restore` if additional rollback is required.

