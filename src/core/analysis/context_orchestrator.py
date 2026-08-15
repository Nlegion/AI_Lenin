"""Context retrieval orchestration for analysis generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.core.analysis.dialectical_config import (
    DialecticalOrchestrationConfig,
    load_dialectical_config,
)
from src.core.analysis.evidence_brief import EvidenceBrief
from src.core.analysis.evidence_brief_builder import build_evidence_brief
from src.core.analysis.semantic_core_config import (
    SemanticCoreConfig,
    load_semantic_core_config,
)

logger = logging.getLogger(__name__)


class AnalysisContextOrchestrator:
    def __init__(
        self,
        retrieval_provider: Any | None,
        *,
        dialectical_config: DialecticalOrchestrationConfig | None = None,
        config_path: Path | None = None,
        taxonomy_path: Path | None = None,
        semantic_config: SemanticCoreConfig | None = None,
        semantic_config_path: Path | None = None,
    ):
        self.retrieval_provider = retrieval_provider
        if dialectical_config is not None:
            self.dialectical_config = dialectical_config
        elif config_path is not None:
            self.dialectical_config = load_dialectical_config(config_path=config_path)
        else:
            self.dialectical_config = DialecticalOrchestrationConfig()
        self.taxonomy_path = taxonomy_path
        if semantic_config is not None:
            self.semantic_config = semantic_config
        else:
            self.semantic_config = load_semantic_core_config(path=semantic_config_path)

    def build_context(self, enhanced_query: str) -> str:
        return self._from_provider(enhanced_query=enhanced_query)

    def build_evidence_brief(
        self,
        *,
        news_title: str,
        news_content: str,
        key_concepts: list[str],
        enhanced_query: str | None = None,
        run_id: str | None = None,
    ) -> EvidenceBrief:
        return build_evidence_brief(
            news_title=news_title,
            news_content=news_content,
            key_concepts=key_concepts,
            enhanced_query=enhanced_query,
            config=self.dialectical_config,
            retrieval_provider=self.retrieval_provider,
            build_context_fn=self.build_context,
            taxonomy_path=self.taxonomy_path,
            semantic_config=self.semantic_config,
            run_id=run_id,
            dialectical_enabled=self.dialectical_config.enabled,
        )

    def _from_provider(self, enhanced_query: str) -> str:
        if self.retrieval_provider is None:
            return ""
        try:
            retrieval_result = self.retrieval_provider.retrieve_context(
                query_text=enhanced_query,
                author_filter="Ленин",
            )
            return retrieval_result.context
        except Exception as error:  # noqa: BLE001
            logger.error("Error in retrieval provider: %s", error)
            return ""
