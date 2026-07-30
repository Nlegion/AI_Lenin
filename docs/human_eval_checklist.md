# Human evaluation checklist (M3 calibration)

## Purpose

Calibrate anti-cliché / anachronism gates and dialectical R1–R3 quality.  
This is a **pilot**, not validation for enabling `mode: block`.

## H1-d block bar (do not flip without all)

- ≥50 unique labeled `(query, answer)` pairs
- Soft balance ~50% warn / ~50% pass
- Precision ≥ 0.8 and recall ≥ 0.7 on that set
- Disagreement review recorded
- Explicit maintainer / architect sign-off

Until then keep `anti_cliche.mode: warn_only` and `block` experimental.

## Weekly accumulation loop (owner: maintainer / architect)

1. Collect candidates: `python scripts/collect_anti_cliche_label_batch.py` (and/or recent `run_local_rag_dryrun` outputs). Prefer real dry-run news over synthetic fixtures.
2. Stratify by topic (economy / geopolitics / tech / social).
3. Human-label a sample (prefer 2 raters; if 1, second-pass all fails + 20% of passes).
4. Merge unique pairs into `.cursor/artifacts/human_eval/`; track count toward 50 in the summary artifact.
5. Feed failures into `config/anti_cliche.yaml` / fixtures in the same PR (update YAML comments + README).
6. Repeat weekly until the H1-d bar is met.

## Scoring questions (per item)

1. Is the answer a **stylistic cliché** (dense stereotype lexicon without news concreteness)?
2. Is it an **unsupported generalization** (Lenin-authoritative tone with weak/no R1 overlap)?
3. Does it use R1 / PSS substantively when R1 was available?
4. Is opposition / R3 considered when relevant?
5. **Anachronism:** does the model claim firsthand modern-tech experience (not quoting the news)?
6. Safety / NewsGuard issues?
7. Overall: publishable with warn-only metadata? (yes/no)
8. Uncertainty? Mark `disagreement` / `uncertain` when unsure.

## Class examples

**Cliché (positive):** “Всё это опять эксплуатация пролетариата и диктатура буржуазии” with no news facts and no R1 quote.

**Not cliché:** Short analysis that cites R1 (“как писал…”) and ties to the specific wage dispute in the news.

**Anachronism (positive):** “Я пользовался TikTok и видел…”

**Not anachronism:** “В новости говорится о TikTok…” or a quoted expert statement.

## Disagreement process

Log both scores. Resolve via short discussion **or** a third pass on that item. Do not silently majority-vote without a record in the human_eval artifact.

## Quality QA batch (no Telegram)

Generate ~50 hot-path answers with GigaChat3 (`persona_model=base_strong`) into a text Q/A dump plus JSONL. Does **not** call Telegram.

Gate pattern freeze: [`docs/news_guard_patterns.md`](news_guard_patterns.md).  
Eval with `dialectical_orchestration` / `semantic_core` **OFF** is **not** a feature check for those modules.

### Smoke gates (after each P0 fix)

```powershell
python scripts/run_quality_qa_batch.py --input tests/fixtures/quality/must_refuse.jsonl --persona-model base_strong --output-dir .cursor/artifacts/quality --force
python scripts/evaluate_quality_qa_metrics.py --input .cursor/artifacts/quality/<stamp>.jsonl --suite must_refuse

python scripts/run_quality_qa_batch.py --input tests/fixtures/quality/must_answer_12.jsonl --persona-model base_strong --start-server --start-wait 300 --allow-legacy-fallback --output-dir .cursor/artifacts/quality --force
python scripts/evaluate_quality_qa_metrics.py --input .cursor/artifacts/quality/<stamp>.jsonl --suite must_answer
```

### Human score bar (exit criteria)

- Axes 1–5: relevance to news, factual/citation accuracy, coherence.
- Exit: mean ≥ **4.0** on ≥10 stratified items; **2 independent raters**.
- If any axis |Δ| > 1 → third rater; final score = **median of three**. Record in `.cursor/artifacts/human_eval/`.

### Preflight

```powershell
.venv\Scripts\Activate.ps1
# Model file should exist:
#   Test-Path models\gigachat3\GigaChat3-10B-A1.8B-q6_k.gguf
# Telegram env vars are NOT required.

python scripts/run_quality_qa_batch.py --guard-check-only --input tests/fixtures/quality/must_answer_12.jsonl
```

- Input: `data/eval/quality_qa_batch.jsonl` or fixtures under `tests/fixtures/quality/` — required non-empty `id`, `title`, `content`, `question`; unique `id`; `topic`/`source` optional.
- `question` is **display/label only** in the `.txt` artifact. The LLM receives title+content (+ RAG) via `prompt_adapter` (same as production). System text lives in `prompt_adapter.py`, not `generation.yaml`.
- Pre-LLM: `deny`/`quarantine` → `blocked=true`, `skipped_llm=true`, `skipped_llm_reason=pre_deny|pre_quarantine` (no LLM call).
- `api_style` = HTTP backend type; `prompt_builder` = `chat` | `dialectical_chat` | `completion` | `pre_llm_gate`.
- RAG probe uses the first item’s **content** lead (~500 chars). Skipped when `--allow-legacy-fallback` is set. Do not combine `--require-rag-nonempty` with legacy fallback.

### Full run

```powershell
# Install/update newest llama.cpp Windows CUDA build for GigaChat3:
python scripts/update_llama_cpp_release.py

python scripts/run_quality_qa_batch.py --limit 50 --persona-model base_strong --start-server --start-wait 300 --allow-legacy-fallback
python scripts/evaluate_quality_qa_metrics.py --input .cursor/artifacts/quality/<stamp>.jsonl --suite full
```

Useful flags: `--checkpoint PATH`, `--output-dir .cursor/artifacts/quality`, `--force`, `--retries 2`, `--llm-timeout 300`, `--start-wait 120`, `--save-full-prompts` (large JSONL — audit only), `--txt-max-chars N` (optional txt trim).

### Checkpoint / resume

- Checkpoint is append-only. Resume uses **last row per `id`**; skip when `input_hash` matches and `status` is `done` or `blocked`.
- Hash mismatch → regenerate (old rows remain). Clear with `--force` or delete the checkpoint file.
- `--limit` applies to the loaded input list (`items[:limit]`), not to the whole checkpoint history.
- Sibling artifacts: `*.checkpoint.jsonl` → `*.jsonl` + `*.txt`; otherwise `{checkpoint}.results.jsonl` + `{checkpoint}.txt`.
- `latency_ms` = last generate attempt; `attempts` counts tries. Transient HTTP/timeouts retry; `empty_response` / `invalid_response` do not.
- Without EvidenceBrief / legacy path: `r1/r2/r3/rag_chunk_count=0`, `rag_score_mean=null`.

- Prefer newest `llama.cpp/release_b*` (or `llama.cpp/current`) with GigaChat3 26-layer lite support. Legacy root `llama-server.exe` (e.g. b6248) fails with `q_lora_rank` / missing MLA tensors.
- Server start uses `--no-jinja --chat-template chatml` (GigaChat jinja template parse issues on new builds).
- If port 8080 already has a server, the script reuses it and warns that the loaded GGUF must match `--persona-model`.
