# AI_Lenin

Local news-RAG pipeline (Telegram): fetch news → retrieve (Qdrant hybrid) → optional dialectical R1–R3 EvidenceBrief → LLM analysis → NewsGuard + anti-cliché / anti-anachronism / lacuna-hedge warn gates.

## Hot path

```text
news → context / EvidenceBrief (R1–R3 when dialectical_orchestration.enabled)
    → generate → text_cleaner → mark_unverified_facts
    → cliche_gate (non-mutating) → anachronism_gate (non-mutating)
    → lacuna_hedge_gate (non-mutating, when semantic_core enabled)
    → NewsGuard.guard_output → publish
```

Stance layers in Qdrant payload `stance_type`:

| Slot | stance_type | Role |
|------|-------------|------|
| R1 | `core_self` | Lenin / PSS |
| R2 | `influence_agree` | supports / agreements |
| R3 | `influence_critical` | opposition / critique |

Details: [`docs/dialectical_orchestration_r1_r3.md`](docs/dialectical_orchestration_r1_r3.md). Priorities: [`docs/priority_crisis_recovery_and_hardening.md`](docs/priority_crisis_recovery_and_hardening.md). Docs index: [`docs/README.md`](docs/README.md). Agent conventions: [`AGENTS.md`](AGENTS.md).

## Generation backend

Default persona is **GigaChat3** (`persona_model: base_strong` in [`config/generation.yaml`](config/generation.yaml)):

| Key | Value |
|-----|--------|
| Model | `ai-sage/GigaChat3-10B-A1.8B` |
| GGUF | `models/gigachat3/GigaChat3-10B-A1.8B-q6_k.gguf` |
| API | OpenAI-compatible `/v1/chat/completions` via local `llama-server` (`http://127.0.0.1:8080`) |
| Prompts | [`src/core/generation/prompt_adapter.py`](src/core/generation/prompt_adapter.py) |

`fine_tuned` (Saiga) remains an optional / legacy fallback backend. Prefer a recent `llama.cpp` Windows CUDA build (`llama.cpp/release_b*` or `llama.cpp/current`); update with:

```powershell
python scripts/update_llama_cpp_release.py
```

Server start for GigaChat3 uses `--no-jinja --chat-template chatml`.

## Setup (short)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Required env (Telegram publish path): `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHANNEL_ID`, `TELEGRAM_ADMIN_ID`.  
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
| `semantic_core` in `config/semantic_core.yaml` | `enabled: false` | Modern→Lenin abstract topic bridge; see [`docs/semantic_core.md`](docs/semantic_core.md) |
| `mode` in `config/anti_cliche.yaml` | `warn_only` | Cliché gate; `block` only after H1-d bar |
| `config/release_gates.yaml` | versioned SoT | RAG thresholds + which release gates run |

`semantic_core` never auto-enables dialectical orchestration. When both are on, abstract slot queries replace raw modern surface terms for R1–R3 retrieval.

## Quality QA batch (no Telegram)

Offline ~50-item hot-path dump for human review. **Does not publish to Telegram** and does not need Telegram env vars.

```powershell
# Preflight (NewsGuard on the eval set)
python scripts/run_quality_qa_batch.py --guard-check-only

# Full run (GigaChat3)
python scripts/run_quality_qa_batch.py --limit 50 --persona-model base_strong --start-server --start-wait 300 --allow-legacy-fallback --output-dir .cursor/artifacts/quality
```

- Input: `data/eval/quality_qa_batch.jsonl` (under gitignored `/data/`; regenerate via `python scripts/_gen_quality_qa_dataset.py` when missing).
- Required fields: non-empty `id`, `title`, `content`, `question`; unique `id`. Optional: `topic`, `source`.
- `question` is **display/label only** in the `.txt` artifact. The LLM receives title+content (+ RAG), same as production.
- Checkpoint is append-only with **last-wins** resume per `id` + `input_hash`. Use `--checkpoint PATH` to continue; `--force` to redo.
- Outputs (siblings): `quality_qa_batch_<stamp>.txt`, `.jsonl`, `.checkpoint.jsonl` under `.cursor/artifacts/quality/`.

Full checklist and flags: [`docs/human_eval_checklist.md`](docs/human_eval_checklist.md).

## Quality / release commands

```powershell
python scripts/run_local_rag_dryrun.py --fixture economy --verbose
python scripts/evaluate_rag_quality.py
python scripts/evaluate_news_guard.py --out-json .cursor/artifacts/evaluation/news_guard_eval.json
python scripts/evaluate_anti_cliche.py
python scripts/release_pass.py --help
python scripts/collect_anti_cliche_label_batch.py
python scripts/calibrate_semantic_core_query.py
python scripts/evaluate_semantic_core.py
python scripts/run_quality_qa_batch.py --guard-check-only
```

`release_pass` CLI flags **override/supplement** `config/release_gates.yaml`:

- `--skip-rag-quality`, `--skip-security-m`, `--skip-anti-cliche`
- `--override-rag-quality REASON` (logs under `.cursor/artifacts/evaluation/`)
- `--check-news-guard-delta` (bootstraps baseline if missing)

## Gates metadata (warn_only)

Cliché / anachronism / lacuna-hedge gates **do not modify** analysis text. They write:

- `metadata["cliche_gate"]`, `metadata["anachronism_gate"]`, `metadata["lacuna_hedge_warn"]`
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

### Lacuna hedge

When `semantic_core` is enabled: warn-only if the model hedges around an empty/exhausted abstract core. Details: [`docs/semantic_core.md`](docs/semantic_core.md).

## Artifact layout

| Dir | Contents |
|-----|----------|
| `.cursor/artifacts/evaluation/` | Automatic eval reports (RAG, anti-cliché, NewsGuard) |
| `.cursor/artifacts/human_eval/` | Human label batches / pilot scores |
| `.cursor/artifacts/quality/` | Quality QA batch dumps (`.txt` / `.jsonl` / checkpoint) |
| `.cursor/artifacts/safety/` | Warn audits / dry-run audit logs |
