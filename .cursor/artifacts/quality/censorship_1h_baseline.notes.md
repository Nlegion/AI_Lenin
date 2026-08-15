# Censorship Isolated Run
- started_utc: 2026-08-09T14:32:37.775575+00:00
- duration_hours: 1.0
- rows: 1440
- poll_seconds: 300
- jsonl: censorship_1h_baseline.jsonl
- csv: censorship_1h_baseline.csv
- metrics: censorship_1h_baseline.metrics.json
- control_path: .cursor/artifacts/quality/censorship_control_set_latest.jsonl
- control_rows_total: 1500
- control_batch_size: 20
- control_cursor_final: 240
- control_exhausted: False
- sidecar: censorship_1h_baseline_allow_bodies.jsonl
- config_version_hash_last: 44e4c006eaed907e
- git_head: b5c4df241c4a8057cfcf283d4eeeaaa95a211aed
- python_version: 3.12.9

## Generation Eval
- qa_input: censorship_1h_allow_qa.jsonl
- qa_results: censorship_1h_allow_qa_20260809-1833.jsonl
- qa_metrics: censorship_1h_allow_qa_metrics.json
- persona_model: base_strong
- persona_model_path: models/gigachat3/GigaChat3-10B-A1.8B-q6_k.gguf
- allow_legacy_fallback: true (RAG fallback flag only)

## Metric Definitions
- refusal_phrase_rate: share of non-skipped LLM answers containing refusal marker from `scripts/evaluate_quality_qa_metrics.py`.
- frg_artifact_rate: share of non-skipped answers containing `ФРГ` artifact marker.
- truncated_marker_rate: share of non-skipped answers containing `[truncated]`.
- reasoning_connector_rate: share with reasoning connectors from `src/core/safety/batch_metrics.py::depth_quality_proxies`.
- lexical_diversity: average unique-token ratio from `depth_quality_proxies`.
- template_phrase_rate: share containing fallback template hints from `depth_quality_proxies`.
- fact_anchor_rate: share with overlap in numbers/proper nouns with source text from `depth_quality_proxies`.