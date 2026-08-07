from pathlib import Path

from src.core.retrieval.provider_factory import load_retrieval_pipeline_config


def test_load_retrieval_pipeline_config_reads_migration_section(tmp_path: Path):
    payload = """
retrieval_pipeline:
  enabled: true
  collection_name: "philosophy_ontology_v2"
  qdrant_path: "database/qdrant_local"
  dense_model: "models/Giga-Embeddings-instruct"
  sparse_state_path: ".cursor/artifacts/qdrant/sparse_encoder_state.json"
  ontology_tags_path: ".cursor/artifacts/ontology/ontology_tags.tsv"
  migration:
    mode: "qdrant_only"
"""
    config_path = tmp_path / "retrieval_pipeline.yaml"
    config_path.write_text(payload, encoding="utf-8")

    config = load_retrieval_pipeline_config(config_path=config_path)
    assert config.migration.mode == "qdrant_only"
