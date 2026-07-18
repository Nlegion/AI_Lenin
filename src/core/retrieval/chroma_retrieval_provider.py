"""Legacy Chroma retrieval provider wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.rag_system import EnhancedRAGSystem
from src.core.retrieval.base_provider import RetrievalProvider, RetrievalResult


@dataclass(frozen=True)
class ChromaRetrievalConfig:
    top_k: int = 7


class ChromaRetrievalProvider(RetrievalProvider):
    def __init__(self, rag_system: EnhancedRAGSystem, config: ChromaRetrievalConfig | None = None):
        self.rag_system = rag_system
        self.config = config or ChromaRetrievalConfig()

    def retrieve_context(self, query_text: str, author_filter: str | None = None) -> RetrievalResult:
        context = self.rag_system.retrieve_relevant_context(
            query=query_text,
            k=self.config.top_k,
            author_filter=author_filter,
        )
        return RetrievalResult(
            context=context,
            candidates_count=0 if not context else len([part for part in context.split("\n\n") if part.strip()]),
            metadata={"provider": "chroma"},
        )
