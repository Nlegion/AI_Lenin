"""RAG preflight fail-fast tests (mocked Qdrant; no Docker)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.core.retrieval.rag_preflight import (
    RagPreflightError,
    run_rag_preflight,
    verify_embedding_repo,
    verify_qdrant_collection,
)


def _write_pipeline_config(
    path: Path, *, qdrant_path: str = "database/qdrant_local"
) -> None:
    payload = {
        "retrieval_pipeline": {
            "enabled": True,
            "collection_name": "philosophy_ontology_giga_v1",
            "qdrant_path": qdrant_path,
            "dense_model": "models/Giga-Embeddings-instruct",
            "sparse_state_path": ".cursor/artifacts/qdrant/sparse_encoder_state_giga_v1.json",
            "ontology_tags_path": ".cursor/artifacts/ontology/ontology_tags.tsv",
            "expected_dim": 2048,
            "migration": {"mode": "qdrant_only"},
        }
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_verify_embedding_repo_requires_files(tmp_path: Path):
    model_dir = tmp_path / "models" / "Giga-Embeddings-instruct"
    model_dir.mkdir(parents=True)
    errors = verify_embedding_repo(model_dir=model_dir)
    assert any("config.json" in item for item in errors)


def test_verify_embedding_repo_ok(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in (
        "config.json",
        "modules.json",
        "config_sentence_transformers.json",
    ):
        (model_dir / name).write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")
    assert verify_embedding_repo(model_dir=model_dir) == []


def test_verify_qdrant_collection_empty_fails(tmp_path: Path):
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    collection = MagicMock()
    collection.name = "philosophy_ontology_giga_v1"
    info = MagicMock()
    info.points_count = 0
    info.config.params.vectors = MagicMock(size=2048)
    client = MagicMock()
    client.get_collections.return_value.collections = [collection]
    client.get_collection.return_value = info
    with patch(
        "src.core.retrieval.rag_preflight.QdrantClient",
        return_value=client,
    ):
        errors = verify_qdrant_collection(
            qdrant_path=qdrant_path,
            collection_name="philosophy_ontology_giga_v1",
            expected_dim=2048,
        )
    assert any("empty" in item for item in errors)


def test_run_rag_preflight_fail_fast(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_pipeline_config(config_dir / "retrieval_pipeline.yaml")
    with pytest.raises(RagPreflightError):
        run_rag_preflight(base_dir=tmp_path)


def test_run_rag_preflight_ok(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_pipeline_config(config_dir / "retrieval_pipeline.yaml")

    model_dir = tmp_path / "models" / "Giga-Embeddings-instruct"
    model_dir.mkdir(parents=True)
    for name in (
        "config.json",
        "modules.json",
        "config_sentence_transformers.json",
    ):
        (model_dir / name).write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")

    sparse = (
        tmp_path
        / ".cursor"
        / "artifacts"
        / "qdrant"
        / "sparse_encoder_state_giga_v1.json"
    )
    sparse.parent.mkdir(parents=True)
    sparse.write_text("{}", encoding="utf-8")
    ontology = tmp_path / ".cursor" / "artifacts" / "ontology" / "ontology_tags.tsv"
    ontology.parent.mkdir(parents=True)
    ontology.write_text("tag\n", encoding="utf-8")
    qdrant_path = tmp_path / "database" / "qdrant_local"
    qdrant_path.mkdir(parents=True)

    collection = MagicMock()
    collection.name = "philosophy_ontology_giga_v1"
    dense = MagicMock()
    dense.size = 2048
    info = MagicMock()
    info.points_count = 10
    info.config.params.vectors = {"dense": dense}
    client = MagicMock()
    client.get_collections.return_value.collections = [collection]
    client.get_collection.return_value = info

    with patch(
        "src.core.retrieval.rag_preflight.QdrantClient",
        return_value=client,
    ):
        run_rag_preflight(base_dir=tmp_path)
