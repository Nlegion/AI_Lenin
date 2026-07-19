"""Factory for retrieval migration modes and provider selection."""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field
import yaml

from src.core.rag_system import EnhancedRAGSystem
from src.core.retrieval.chroma_retrieval_provider import ChromaRetrievalConfig, ChromaRetrievalProvider
from src.core.retrieval.migration_provider import MigrationConfig, MigrationRetrievalProvider
from src.core.retrieval.qdrant_retrieval_provider import QdrantRetrievalProvider, RetrievalProviderConfig


class MigrationSection(BaseModel):
    mode: str = "qdrant_only"
    parity_min_shared_ratio: float = 0.25
    audit_log_path: str = ".cursor/artifacts/retrieval/retrieval_ab_audit.jsonl"


class RetrievalPipelineConfig(BaseModel):
    enabled: bool = True
    collection_name: str
    qdrant_path: str
    dense_model: str
    sparse_state_path: str
    ontology_tags_path: str
    trust_remote_code: bool = False
    device: str = "auto"
    fallback_to_cpu: bool = True
    expected_dim: int | None = 2048
    top_k: int = 20
    rrf_k: int = 60
    max_context_chunks: int = 7
    hyde_enabled: bool = False
    query_rewrite_enabled: bool = True
    query_decomposition_enabled: bool = False
    judge_enabled: bool = True
    judge_alpha: float = 0.2
    retriever_weights: dict[str, float] = Field(default_factory=dict)
    source_boosts: dict[str, float] = Field(default_factory=dict)
    migration: MigrationSection = Field(default_factory=MigrationSection)
    chroma_top_k: int = 7


def load_retrieval_pipeline_config(config_path: Path) -> RetrievalPipelineConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("retrieval_pipeline", payload)
    return RetrievalPipelineConfig.model_validate(section)


def build_provider(
    config_path: Path,
    base_dir: Path,
    rag_system: EnhancedRAGSystem | None,
):
    if not config_path.exists():
        return None
    config = load_retrieval_pipeline_config(config_path=config_path)
    if not config.enabled:
        return None

    dense_model = config.dense_model
    local_dense = base_dir / dense_model
    if local_dense.exists():
        dense_model = str(local_dense.resolve())

    qdrant_provider = QdrantRetrievalProvider(
        config=RetrievalProviderConfig(
            collection_name=config.collection_name,
            qdrant_path=base_dir / config.qdrant_path,
            dense_model=dense_model,
            sparse_state_path=base_dir / config.sparse_state_path,
            ontology_tags_path=base_dir / config.ontology_tags_path,
            trust_remote_code=config.trust_remote_code,
            device=config.device,
            fallback_to_cpu=config.fallback_to_cpu,
            expected_dim=config.expected_dim,
            top_k=config.top_k,
            rrf_k=config.rrf_k,
            retriever_weights=config.retriever_weights,
            source_boosts=config.source_boosts,
            max_context_chunks=config.max_context_chunks,
            hyde_enabled=config.hyde_enabled,
            query_rewrite_enabled=config.query_rewrite_enabled,
            query_decomposition_enabled=config.query_decomposition_enabled,
            judge_enabled=config.judge_enabled,
            judge_alpha=config.judge_alpha,
        )
    )
    migration_mode = config.migration.mode
    if migration_mode == "qdrant_only":
        return qdrant_provider

    if rag_system is None:
        return qdrant_provider

    chroma_provider = ChromaRetrievalProvider(
        rag_system=rag_system,
        config=ChromaRetrievalConfig(top_k=config.chroma_top_k),
    )
    if migration_mode == "chroma_only":
        return chroma_provider
    if migration_mode == "ab_shadow":
        return MigrationRetrievalProvider(
            primary=qdrant_provider,
            shadow=chroma_provider,
            config=MigrationConfig(
                mode="ab_shadow",
                parity_min_shared_ratio=config.migration.parity_min_shared_ratio,
                audit_log_path=base_dir / config.migration.audit_log_path,
            ),
            primary_name="qdrant",
            shadow_name="chroma",
        )
    raise ValueError(f"Unsupported retrieval migration mode: {migration_mode}")
