from pathlib import Path

from src.core.retrieval.provider_factory import load_retrieval_pipeline_config


def test_load_retrieval_pipeline_config_reads_migration_section(tmp_path: Path):
    payload = """
retrieval_pipeline:
  enabled: true
  collection_name: "philosophy_ontology_v2"
  qdrant_path: "database/qdrant_local"
  dense_model: "sentence-transformers/all-MiniLM-L6-v2"
  sparse_state_path: ".cursor/artifacts/qdrant/sparse_encoder_state.json"
  ontology_tags_path: ".cursor/artifacts/ontology/ontology_tags.tsv"
  migration:
    mode: "ab_shadow"
    parity_min_shared_ratio: 0.4
    audit_log_path: ".cursor/artifacts/retrieval/retrieval_ab_audit.jsonl"
"""
    config_path = tmp_path / "retrieval_pipeline.yaml"
    config_path.write_text(payload, encoding="utf-8")

    config = load_retrieval_pipeline_config(config_path=config_path)
    assert config.migration.mode == "ab_shadow"
    assert config.migration.parity_min_shared_ratio == 0.4
