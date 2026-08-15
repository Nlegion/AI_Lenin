# SafetyGate Rule-Authoring Guide

SafetyGate is the pre-LLM censorship component. Generation quality hooks must not
duplicate policy deny/skip decisions.

## Files

| Path | Role |
|------|------|
| `src/core/safety/safety_gate.py` | Gate evaluation + shadow dual-run |
| `src/core/safety/safety_gate_types.py` | `GateContext`, `GateDecision`, `SafetyHint` |
| `src/core/settings/safety_gate_config.py` | Config loader + news_guard key fallback |
| `config/safety_gate_config.yaml` | Policy + feature flags (primary SoT) |
| `scripts/safety/migrate_news_guard_to_safety_gate.py` | One-shot policy migration |

## Feature flags

```yaml
safety_gate:
  flags:
    enabled: true
    shadow_mode: true      # dual-run; enforce legacy until parity proven
    enforce_mode: old      # old|new
    async_shadow: false
    cache_enabled: true
    fallback_to_news_guard_keys: true
```

Rollback is config-only: set `enabled: false` or `enforce_mode: old` and restart.

## Adding a rule

1. Add a private method `_rule_<name>(self, ctx: GateContext) -> RuleResult`.
2. Return `RuleResult(hit=False)` when the rule does not apply.
3. Do **not** add new business rules to `news_guard.py` after SafetyGate bootstrap.
4. Critical bug-fixes in legacy helpers are allowed only if mirrored into SafetyGate
   in the same change and covered by parity tests.
5. Prefer typed `SafetyHint` values for prompt modifiers; never put prompt text into
   `trace` (trace is debug-only).

## Yellow handling

- Pre-LLM: `risk_tier=yellow` + `context_hints` (`YELLOW_CONSTRAINED_ANALYSIS`, …).
- Prompt: `prompt_adapter` appends constrained-analysis paragraph from hints.
- Upstream warning text may be injected into analysis body (`needs_yellow_warning`).
- Publisher remains transport-only (Telegram formatting).

## Parity / red suite

- Overall shadow parity target: >= 95%.
- Red-deny critical parity: >= 99%; red leakage must be 0% in the red suite.
- Stratify samples: red military, yellow economy-sensitive, neutral allow, sport skip, FIO.

## Migration

```powershell
.venv\Scripts\python.exe scripts/safety/migrate_news_guard_to_safety_gate.py
```

Missing keys in `safety_gate_config.yaml` temporarily fall back to `news_guard.yaml`
with warning logs; fallback usage must reach zero before Stage 3 completion.
