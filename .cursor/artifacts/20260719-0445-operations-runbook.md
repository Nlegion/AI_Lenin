# Operations Runbook

## 1. Full Index Rebuild From Scratch
1. Bootstrap environment:
   - `python scripts/bootstrap_dev_env.py`
2. Cleanup stale state:
   - `python scripts/cleanup_pre_rerun_state.py`
3. Rebuild cleaned corpus:
   - `python scripts/rebuild_clean_corpus.py --registry .cursor/artifacts/registries/source_registry.tsv --corpus-root data/books --config config/cleaning_rules.yaml --cleaned-root .cursor/artifacts/cleaned_corpus --qa-json .cursor/artifacts/cleaning/cleaning_qa.json --summary-md .cursor/artifacts/cleaning/cleaning_summary.md`
4. Rebuild chunk dataset:
   - `python scripts/build_chunk_dataset_v2.py --registry .cursor/artifacts/registries/source_registry.tsv --cleaned-root .cursor/artifacts/cleaned_corpus --config config/chunking_rules.yaml --chunks-output .cursor/artifacts/chunks/chunk_dataset_v2.tsv --summary-output .cursor/artifacts/chunks/chunking_summary.md --qa-output .cursor/artifacts/chunks/chunking_qa.json`
5. Rebuild ontology tags/graph:
   - `python scripts/build_ontology_worldview.py --registry .cursor/artifacts/registries/source_registry.tsv --corpus-root data/books --taxonomy-config config/ontology_taxonomy.yaml --tags-output .cursor/artifacts/ontology/ontology_tags.tsv --graph-output .cursor/artifacts/ontology/worldview_graph.json --validation-output .cursor/artifacts/ontology/validation_sample.tsv --summary-output .cursor/artifacts/ontology/ontology_summary.md`
6. Full ingestion to Qdrant:
   - `python scripts/build_qdrant_index.py --config config/qdrant_ingestion.yaml --chunks-tsv .cursor/artifacts/chunks/chunk_dataset_v2.tsv --stats-json .cursor/artifacts/qdrant/ingestion_stats.json --summary-md .cursor/artifacts/qdrant/ingestion_summary.md`
7. Evaluate retrieval:
   - `python scripts/run_retrieval_sandbox.py --config config/retrieval_sandbox.yaml --out-json .cursor/artifacts/sandbox/retrieval_sandbox_results.json --out-md .cursor/artifacts/sandbox/retrieval_sandbox_summary.md --max-queries 120`
   - `python scripts/evaluate_rag_quality.py --retrieval-config config/retrieval_pipeline.yaml --thresholds-config config/quality_thresholds.yaml --eval-dataset .cursor/artifacts/eval/embedding_eval.tsv --output-json .cursor/artifacts/evaluation/rag_quality_metrics.json --output-md .cursor/artifacts/evaluation/rag_quality_summary.md --max-queries 120`

## 2. Migration Mode Switching
- Config file: `config/retrieval_pipeline.yaml`
- Modes:
  - `ab_shadow`
  - `qdrant_only`
  - `chroma_only`
- Validate mode health:
  - `python scripts/run_retrieval_ab_check.py --config config/retrieval_pipeline.yaml --out-json .cursor/artifacts/retrieval/retrieval_ab_summary.json --out-md .cursor/artifacts/retrieval/retrieval_ab_summary.md`

## 3. Recovery After Qdrant Failure
1. Stop runtime entrypoint.
2. Remove possibly corrupted local storage:
   - `database/qdrant_local`
3. Remove stale ingestion checkpoint:
   - `.cursor/artifacts/qdrant/checkpoints/ingestion.offset`
4. Re-run full ingestion from chunk dataset.
5. Run retrieval sanity checks:
   - sandbox summary,
   - quality summary,
   - A/B parity summary.

## 4. Security/Legal Regression
- NewsGuard regression:
  - `python scripts/evaluate_news_guard.py --config config/news_guard.yaml --out-json .cursor/artifacts/safety/news_guard_eval_release.json --out-md .cursor/artifacts/safety/news_guard_eval_release.md`
- Public disclaimer policy:
  - `python scripts/validate_public_news_policy.py --config config/news_guard.yaml --public-mode`

## 5. Release-pass
- Execute:
  - `python scripts/release_pass.py`
- Current known blocker:
  - legacy lint debt in repository test files may fail ruff gate.

## 6. Local Dry-Run (No Telegram)
- Standard run:
  - `python scripts/run_local_rag_dryrun.py --fixture economy --verbose`
- High-speed retrieval debug:
  - `python scripts/run_local_rag_dryrun.py --fixture politics --skip-judge --verbose`
- Safety fixture checks:
  - `python scripts/run_local_rag_dryrun.py --fixture provocative`
  - `python scripts/run_local_rag_dryrun.py --fixture conflict`
- Audit/alerts:
  - `python scripts/run_local_rag_dryrun.py --fixture provocative --audit-log .cursor/artifacts/safety/dryrun_audit.jsonl --alert-threshold 3`
- Custom text from stdin:
  - `Get-Content .\news.txt | python scripts/run_local_rag_dryrun.py --news-text - --verbose`

## 7. Dry-Run Diagnostics
- `Qdrant connection refused`:
  - Causes: local storage lock, corrupted local index, missing collection.
  - Actions: stop parallel processes using Qdrant path, rebuild index, verify `collection_name` and `qdrant_path`.
- `empty context`:
  - Causes: weak query expansion, sparse encoder state mismatch, ontology tags path mismatch.
  - Actions: check `RETRIEVAL_*` sections in verbose output, rebuild sparse state and ingestion.
- `NewsGate deny/quarantine`:
  - Causes: military/PII/high-risk trigger or source whitelist mismatch.
  - Actions: inspect `SAFETY` codes, validate input source and policy config, use fixture tests for regression.
- `low core_self_ratio`:
  - Causes: stance boosts misconfigured, weak dense retrieval, ontology mapping drift.
  - Actions: inspect `ARBITER` score tables, tune `source_boosts`, rerun quality evaluation.

## 8. Public-Mode Compliance Checklist
- Ensure public deployments use `safe_mode: strict`.
- Keep mandatory AI disclaimer enabled in every response.
- Maintain owner/legal contact details in public channel/site metadata.
- Keep audit logs for gate decisions and moderation events.
- Monitor repeated high-risk events from `.cursor/artifacts/safety/dryrun_audit.jsonl` and trigger admin response.
- Perform legal review before enabling public publishing.

## 9. Generation Backends (GigaChat3 primary)
- Config: `config/generation.yaml`, registry: `config/model_registry.yaml`
- Default: `persona_model=base_strong` (GigaChat3-10B chat API)
- Optional: `persona_model=fine_tuned` (Saiga Lenin GGUF `/completion`) — non-default reserve/fallback
- Place GGUF at `models/gigachat3/GigaChat3-10B-A1.8B-q6_k.gguf` (copy/junction from `P:\hometest_GigaChat3\data\model\...`)
- Dense embeddings production path: `models/Giga-Embeddings-instruct` (offline; HF id is provenance only)
- Collection: `philosophy_ontology_giga_v1` (production target). `philosophy_ontology_v2` (MiniLM) is obsolete — keep until explicit cleanup approval.
- Smoke embeddings: `python scripts/smoke_giga_embeddings.py --device auto`
- Dry-run:
  - `python scripts/run_local_rag_dryrun.py --fixture economy --persona-model base_strong --verbose --allow-legacy-fallback`
  - `python scripts/compare_generation_backends.py --allow-legacy-fallback`
- Fallback hook: `generation.safety.fallback` (disabled by default; recommends/selects fine_tuned when incident threshold exceeded)
- Exclusive GPU: RTX 4060 8GB cannot hold Giga-Embeddings + GigaChat3 together. If llama-server is up, embeddings fall back to CPU; ingest jobs should stop LLM first.

## 9b. GPU / PyTorch (CUDA)
- Verify driver: `nvidia-smi`
- Install CUDA torch (do not use default PyPI CPU wheel):
  - `pip uninstall torch torchvision torchaudio -y`
  - `pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126`
- Verify: `python -c "import torch; assert torch.cuda.is_available()"`
- Configs use `device: auto` + `fallback_to_cpu: true` (`src/core/settings/device.py`)
- CUDA Toolkit full install is usually unnecessary for torch wheels; only needed if custom CUDA kernels fail to build
- Typical errors:
  - `CUDA out of memory` → reduce ingest `batch_size` 32→16→8→4 (`adaptive_batch`)
  - `no CUDA-capable device` → check driver / reboot; resolver falls back to CPU
- llama.cpp uses its own CUDA runtime and does **not** depend on torch CUDA
- Ingest resume: fingerprint in `.cursor/artifacts/qdrant/checkpoints/ingestion_giga_v1.meta.json`; mismatch requires `--reset-checkpoint`
- Long-run ops check during full ingest: watch `nvidia-smi` temperature and memory for leaks/degradation

## 10. Legal Residual Risks
- GigaChat3 may hallucinate facts; publishing false claims can create legal exposure.
- Residual prohibited-content risk remains despite NewsGate/NewsGuard filters.
- `base_strong` is less corpus-bound than fine-tuned; content responsibility is higher.
- Embedding cutover to Giga-Embeddings requires ideology metrics checks (`core_self_ratio`, `attribution_coverage`).
- Legal review is mandatory before public publishing.
