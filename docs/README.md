# Docs index

Canonical technical SoT. Prefer these over `.cursor/artifacts/` session notes.

## Pipeline / generation

| Document | Role |
|----------|------|
| [answer_postprocess.md](answer_postprocess.md) | Post-generation scrub (`postprocess_clean` pre/post-guard) |
| [quote_grounding.md](quote_grounding.md) | Quote candidates, chunk citation fields, grounding rules |
| [dialectical_orchestration_r1_r3.md](dialectical_orchestration_r1_r3.md) | R1–R3 EvidenceBrief orchestration |
| [dialectical_reasoning_engine.md](dialectical_reasoning_engine.md) | Dialectical reasoning engine (triad/causal QC) |
| [llm_client.md](llm_client.md) | LLM HTTP client / llama-server lifecycle (`src/core/llm/`) |
| [dialectical_r3_data_track.md](dialectical_r3_data_track.md) | R3 corpus / principle-seed coverage track |
| [semantic_core.md](semantic_core.md) | Modern→Lenin abstract topic bridge |

## Safety / censorship

| Document | Role |
|----------|------|
| [news_guard_patterns.md](news_guard_patterns.md) | Input-gate pattern freeze / expansion policy |
| [safety_gate.md](safety_gate.md) | SafetyGate rule-authoring, flags, yellow hints |
| [safety_gate_ops.md](safety_gate_ops.md) | Dual-run ops, rollback switches, dashboard metrics |
| [censor_manual_terms.md](censor_manual_terms.md) | Pre-RAG manual trigger YAML lists / scrub / overrides |
| [censorship_pipeline_wiring.md](censorship_pipeline_wiring.md) | PreRagCensor runtime order vs NewsGuard output guard |
| [censorship_auto_rollback_runbook.md](censorship_auto_rollback_runbook.md) | Shadow rollback when censorship gates degrade |
| [censorship_legacy_decommission_plan.md](censorship_legacy_decommission_plan.md) | Legacy NewsGuard/SafetyGate decommission phases |
| [adr_censorship_dedup_and_external_datasets.md](adr_censorship_dedup_and_external_datasets.md) | Dedup cache + open-license dataset ADR |
| [trial50_hotfix_notes.md](trial50_hotfix_notes.md) | Trial50 safety/generation hotfix flags and verify |

## Quality / ops

| Document | Role |
|----------|------|
| [priority_crisis_recovery_and_hardening.md](priority_crisis_recovery_and_hardening.md) | Crisis recovery priorities H1–H3 + M1–M3 |
| [human_eval_checklist.md](human_eval_checklist.md) | Human eval / weekly label loop / H1-d bar |
| [config_ownership.md](config_ownership.md) | Config SoT / ownership for retrieval vs generation knobs |
| [docker.md](docker.md) | VPS Docker RAG replica (remote LLM seam; no local llama-server) |

Also see root [`README.md`](../README.md) (architecture, business process, RAG principles / data sources), [`AGENTS.md`](../AGENTS.md), [`CHANGELOG.md`](../CHANGELOG.md), and [`scripts/README.md`](../scripts/README.md).
