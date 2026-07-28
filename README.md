# AI_Lenin

Local news-RAG pipeline (Telegram): fetch news → retrieve (Qdrant hybrid) → optional dialectical R1–R3 EvidenceBrief → LLM analysis → NewsGuard + anti-cliché / anti-anachronism warn gates.

## Hot path

```text
news → context / EvidenceBrief (R1–R3 when dialectical_orchestration.enabled)
    → generate → text_cleaner → mark_unverified_facts
    → cliche_gate (non-mutating) → anachronism_gate (non-mutating)
    → NewsGuard.guard_output → publish
```

Stance layers in Qdrant payload `stance_type`:

| Slot | stance_type | Role |
|------|-------------|------|
| R1 | `core_self` | Lenin / PSS |
| R2 | `influence_agree` | supports / agreements |
| R3 | `influence_critical` | opposition / critique |

Details: [`docs/dialectical_orchestration_r1_r3.md`](docs/dialectical_orchestration_r1_r3.md). Priorities: [`docs/priority_crisis_recovery_and_hardening.md`](docs/priority_crisis_recovery_and_hardening.md). Docs index: [`docs/README.md`](docs/README.md). Agent conventions: [`AGENTS.md`](AGENTS.md).

## Setup (short)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Required env: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHANNEL_ID`, `TELEGRAM_ADMIN_ID`.  
Run app: `python src/main.py`. Local LLM via llama-server / configured backend in `config/generation.yaml`.

### Qdrant stance index

Requires **qdrant-client >= 1.7.0**. Once per environment DB:

```powershell
.\.venv\Scripts\python.exe scripts/ensure_qdrant_stance_index.py
```

## Feature flags

| Flag / config | Default | Meaning |
|---------------|---------|---------|
| `dialectical_orchestration` in `config/retrieval_pipeline.yaml` | `enabled: false` | R1–R3 EvidenceBrief hot path |
| `mode` in `config/anti_cliche.yaml` | `warn_only` | Cliché gate; `block` only after H1-d bar |
| `config/release_gates.yaml` | versioned SoT | RAG thresholds + which release gates run |

## Quality / release commands

```powershell
python scripts/run_local_rag_dryrun.py --fixture economy --verbose
python scripts/evaluate_rag_quality.py
python scripts/evaluate_news_guard.py --out-json .cursor/artifacts/evaluation/news_guard_eval.json
python scripts/evaluate_anti_cliche.py
python scripts/release_pass.py --help
python scripts/collect_anti_cliche_label_batch.py
```

`release_pass` CLI flags **override/supplement** `config/release_gates.yaml`:

- `--skip-rag-quality`, `--skip-security-m`, `--skip-anti-cliche`
- `--override-rag-quality REASON` (logs under `.cursor/artifacts/evaluation/`)
- `--check-news-guard-delta` (bootstraps baseline if missing)

## Gates metadata (warn_only)

Cliché / anachronism gates **do not modify** analysis text. They write:

- `metadata["cliche_gate"]`, `metadata["anachronism_gate"]`
- `hallucination_codes` remains NewsGuard `mark_unverified_facts` only

Warn events append to `.cursor/artifacts/safety/gate_warn_audit.jsonl` (`GATE_WARN_AUDIT_PATH`).  
Gate errors use `logger.exception` (typically **stderr**); warn rows go to that JSONL.

### Jaccard SoT

All gate Jaccard uses **`src.core.analysis.jaccard_metrics` exclusively** (token length ≥ 3, casefold, `JACCARD_STOPWORDS`). Do not add a second Jaccard implementation. See comment in `config/anti_cliche.yaml`.

### Interpreting cliché warns

- `cliche_no_r1` — dialectical brief present, empty R1, dense cliché lexicon (`quote_anchor` does **not** clear this)
- `cliche_low_r1_overlap` + `cliche_lexicon_dense` — both may fire together
- `cliche_skipped_no_brief` — info skip when `brief is None` (legacy path)

Actions: inspect R1 retrieval, expand fixtures/lexicon, do **not** flip `block` until human-eval bar (see [`docs/human_eval_checklist.md`](docs/human_eval_checklist.md)).

### Anachronism

Warn only on first-person × modern-tech outside quotes / attribution. Bare tech in the news → pass.

## Artifact layout

| Dir | Contents |
|-----|----------|
| `.cursor/artifacts/evaluation/` | Automatic eval reports (RAG, anti-cliché, NewsGuard) |
| `.cursor/artifacts/human_eval/` | Human label batches / pilot scores |
| `.cursor/artifacts/safety/` | Warn audits / dry-run audit logs |
