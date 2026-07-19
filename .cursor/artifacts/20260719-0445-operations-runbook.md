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
