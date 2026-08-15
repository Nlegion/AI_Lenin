"""Fail-fast RAG asset checks for VPS / remote LLM mode."""

from __future__ import annotations

import logging
from pathlib import Path

from qdrant_client import QdrantClient

from src.core.retrieval.provider_factory import load_retrieval_pipeline_config

logger = logging.getLogger(__name__)

_EMBEDDING_REQUIRED_FILES = (
    "config.json",
    "modules.json",
    "config_sentence_transformers.json",
)


class RagPreflightError(RuntimeError):
    """Raised when required RAG assets are missing or incompatible."""


def dir_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def verify_embedding_repo(model_dir: Path) -> list[str]:
    errors: list[str] = []
    if not model_dir.is_dir():
        return [f"embedding model dir missing: {model_dir}"]
    for name in _EMBEDDING_REQUIRED_FILES:
        if not (model_dir / name).is_file():
            errors.append(f"embedding repo missing file: {model_dir / name}")
    has_weights = (
        any(model_dir.glob("model-*.safetensors"))
        or (model_dir / "model.safetensors").is_file()
        or (model_dir / "pytorch_model.bin").is_file()
    )
    if not has_weights:
        errors.append(f"embedding repo missing weights under: {model_dir}")
    return errors


def verify_qdrant_collection(
    *,
    qdrant_path: Path,
    collection_name: str,
    expected_dim: int | None,
) -> list[str]:
    errors: list[str] = []
    if not qdrant_path.exists():
        return [f"qdrant path missing: {qdrant_path}"]
    client: QdrantClient | None = None
    try:
        client = QdrantClient(path=str(qdrant_path))
        names = {item.name for item in client.get_collections().collections}
        if collection_name not in names:
            return [f"qdrant collection missing: {collection_name}"]
        info = client.get_collection(collection_name=collection_name)
        points = int(getattr(info, "points_count", 0) or 0)
        if points <= 0:
            errors.append(
                f"qdrant collection empty: {collection_name} points_count={points}"
            )
        if expected_dim is not None:
            vectors = getattr(info.config.params, "vectors", None)
            size = None
            if hasattr(vectors, "size"):
                size = int(vectors.size)
            elif isinstance(vectors, dict):
                dense = vectors.get("dense")
                if dense is not None:
                    size = int(getattr(dense, "size", 0) or 0)
            if size is not None and size != expected_dim:
                errors.append(
                    f"qdrant dense dim mismatch: got={size} expected={expected_dim}"
                )
    except Exception as error:  # noqa: BLE001
        errors.append(f"qdrant preflight failed: {error}")
    finally:
        if client is not None:
            client.close()
    return errors


def run_rag_preflight(*, base_dir: Path) -> None:
    """Validate RAG snapshot paths and Qdrant collection; raise on failure."""
    config_path = base_dir / "config" / "retrieval_pipeline.yaml"
    if not config_path.is_file():
        raise RagPreflightError(f"retrieval pipeline config missing: {config_path}")
    config = load_retrieval_pipeline_config(config_path=config_path)
    if not config.enabled:
        raise RagPreflightError("retrieval_pipeline.enabled is false")

    dense_dir = base_dir / config.dense_model
    sparse_path = base_dir / config.sparse_state_path
    ontology_path = base_dir / config.ontology_tags_path
    qdrant_path = base_dir / config.qdrant_path

    errors = verify_embedding_repo(model_dir=dense_dir)
    if not sparse_path.is_file():
        errors.append(f"sparse encoder state missing: {sparse_path}")
    if not ontology_path.is_file():
        errors.append(f"ontology tags missing: {ontology_path}")
    errors.extend(
        verify_qdrant_collection(
            qdrant_path=qdrant_path,
            collection_name=config.collection_name,
            expected_dim=config.expected_dim,
        )
    )
    if errors:
        for item in errors:
            logger.error("rag_preflight_error detail=%s", item)
        raise RagPreflightError("; ".join(errors))

    logger.info(
        "rag_preflight_ok collection=%s embedding_bytes=%s qdrant_bytes=%s",
        config.collection_name,
        dir_size_bytes(dense_dir),
        dir_size_bytes(qdrant_path),
    )
