# Dialectical stance audit (Phase 0)

- Generated: 2026-07-27
- Collection target: `philosophy_ontology_giga_v1`
- qdrant-client minimum: `>=1.7.0` (env observed: 1.18.0)

## Actions

1. Use `scripts/ensure_qdrant_stance_index.py` once per DB to create/wait `stance_type` KEYWORD payload index (not hot path).
2. Calibrate `slot_timeout_sec` / `retrieve_wall_timeout_sec` after filtered vs unfiltered latency probe on local machine.
3. Chunk-layer stance distribution is available from `.cursor/artifacts/evaluation/retrieval_foundations_audit.md` (chunk_dataset_v2).

## Notes

- Local Qdrant path open can take tens of seconds with 50k+ points; prefer off-peak index ensure.
- Embedded/local Qdrant: payload indexes have **no effect** (warning from client); `create_payload_index` is still invoked for API parity, filters use scan.
- Hot path never auto-creates payload indexes.
