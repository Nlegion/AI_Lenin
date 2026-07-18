"""Context retrieval orchestration for analysis generation."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


class AnalysisContextOrchestrator:
    def __init__(self, retrieval_provider: Any | None, rag_system: Any | None):
        self.retrieval_provider = retrieval_provider
        self.rag_system = rag_system

    def build_context(self, enhanced_query: str) -> str:
        context = self._from_provider(enhanced_query=enhanced_query)
        if context:
            return context
        return self._from_legacy_rag(enhanced_query=enhanced_query)

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

    def _from_legacy_rag(self, enhanced_query: str) -> str:
        if self.rag_system is None:
            return ""
        try:
            context = self.rag_system.retrieve_relevant_context(
                query=enhanced_query,
                k=7,
                author_filter="Ленин",
            )
            if len(context.split()) < 150:
                additional_context = self.rag_system.retrieve_relevant_context(
                    query=enhanced_query,
                    k=3,
                    author_filter="МарксЭнгельс",
                )
                if additional_context:
                    context = f"{context}\n\n{additional_context}" if context else additional_context
            return context
        except Exception as error:  # noqa: BLE001
            logger.error("Error in legacy RAG retrieval: %s", error)
            return ""
