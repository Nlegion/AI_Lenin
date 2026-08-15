# Dialectical Reasoning Engine

**Status:** implemented behind `dialectical_reasoning.mode` (default `orchestration_single_pass`).  
**Related:** [dialectical_orchestration_r1_r3.md](dialectical_orchestration_r1_r3.md) = EvidenceBrief R1–R3 retrieve only.

## Modes

| Mode | Behavior |
|------|----------|
| `legacy` | No dialectical brief required (fail-soft via orchestration flag) |
| `orchestration_single_pass` | Current R1–R3 brief + single LLM chat (default) |
| `reasoning_shadow` | Engine runs; live text stays single-pass; sampled JSONL shadow |
| `reasoning_publish` | Engine rendered text is published (when outcome=`publish`) |

Config section: `dialectical_reasoning` in [`config/retrieval_pipeline.yaml`](../config/retrieval_pipeline.yaml).  
`kill_switch: true` forces mode back to `orchestration_single_pass`.

## Package

`src/core/dialectics/` — isolated; depends on `EvidenceBrief` types + `GenerationBackend` Protocol only.

## Outcomes

`publish | hold_review | suppress` + composable `reason_codes` (e.g. `r3_absent`, `parse_error`, `timeout`).

## Quality

- No fake «Механизм: анализ опирается…» stub (`quality_hooks`).
- Analysis labels `Факт`/`Вывод` are not stripped by artifact scaffold.
- Render budget ≤ `MAX_FINAL_ANSWER_CHARS` (1000).
- Extractive PrincipleCards only (`quote ⊆ chunk`).

## Dry-run

```powershell
python scripts/dialectics/run_dialectical_reasoning_dryrun.py --fixture neftegaz
```
