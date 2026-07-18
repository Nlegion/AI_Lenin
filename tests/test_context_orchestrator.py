from src.core.analysis.context_orchestrator import AnalysisContextOrchestrator
from src.core.retrieval.base_provider import RetrievalResult


class _ProviderStub:
    def __init__(self, context: str):
        self.context = context

    def retrieve_context(self, query_text: str, author_filter: str | None = None) -> RetrievalResult:
        _ = (query_text, author_filter)
        return RetrievalResult(context=self.context, candidates_count=1, metadata={"provider": "stub"})


class _RagStub:
    def __init__(self):
        self.calls: list[tuple[str, int, str]] = []

    def retrieve_relevant_context(self, query: str, k: int = 7, author_filter: str | None = None) -> str:
        self.calls.append((query, k, author_filter or ""))
        if author_filter == "Ленин":
            return "краткий контекст"
        return "дополнительный контекст"


def test_provider_context_has_priority_over_legacy_rag():
    provider = _ProviderStub(context="provider context")
    rag = _RagStub()
    orchestrator = AnalysisContextOrchestrator(retrieval_provider=provider, rag_system=rag)
    context = orchestrator.build_context(enhanced_query="query")
    assert context == "provider context"
    assert rag.calls == []


def test_legacy_rag_adds_secondary_author_context_when_short():
    orchestrator = AnalysisContextOrchestrator(retrieval_provider=None, rag_system=_RagStub())
    context = orchestrator.build_context(enhanced_query="query")
    assert "краткий контекст" in context
    assert "дополнительный контекст" in context
