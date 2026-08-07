# Policy duplication audit (Stage 2)

Classification of censorship/policy decisions outside / around generation.

| Location | Behavior | Action |
|----------|----------|--------|
| `processor.py` pre-LLM `SafetyGate` / `NewsGuard.evaluate_input` | deny/skip/quarantine | **keep** in safety layer |
| `post_generate_gates.py` → `guard_output` | extremism/PII/output blocks | **keep** final safety check |
| `processor.py` post-LLM `guard_output` | duplicate moderation | keep for publish path; avoid second yellow pattern injection |
| `output_artifacts.py` `combat_sensitive` deny | policy-like deny | **removed** from quality path (soft mode) |
| `quote_postcheck.py` static template replace | quality→censorship cascade | **converted** to soft strip (`quote_postcheck_enforce_mode=soft`) |
| `loop_detect.py` static_insufficient | template substitution | **converted** to paragraph dedupe |
| `yellow_output_filter_enabled` | post-gen yellow blocks | **disabled** Stage 0; yellow via SafetyGate hints |

Source of truth:
- Stage 0–1: `news_guard.yaml` + shadow SafetyGate
- Stage 3 target: `safety_gate_config.yaml`
