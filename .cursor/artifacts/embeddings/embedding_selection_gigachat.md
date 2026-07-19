# Embedding Selection: Giga-Embeddings-instruct

- Generated at (UTC): `2026-07-19T07:40:26Z`
- Production local_path: `P:\AI_Lenin\models\Giga-Embeddings-instruct`
- Provenance hf_id: `ai-sage/Giga-Embeddings-instruct`
- Torch: `2.8.0+cu126` GPU=`NVIDIA GeForce RTX 4060`
- Resolved device: `cuda` fallback_to_cpu=`True`
- Offline load: `true`

## Smoke Load
- status: `ok`
- embedding_dim: `2048` (expected `2048`)

## Decision
- production_target_configured: `true`
- smoke_load_ok: `true`
- runtime_collection: `philosophy_ontology_giga_v1`
- obsolete_collection: `philosophy_ontology_v2` (MiniLM; keep until explicit cleanup)
- seed_ingest_512: `complete` (cuda, torch=2.8.0+cu126)
- post_seed_cosine_vs_reference: `1.0`
- full_corpus_ingest: `in_progress` (GPU process active; checkpoint advancing past 512; monitor via `nvidia-smi` and `ingestion_giga_v1.offset`)
