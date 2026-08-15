from src.core.analysis.context_orchestrator import AnalysisContextOrchestrator
from src.core.retrieval.base_provider import RetrievalResult


class _ProviderStub:
    def __init__(self, context: str):
        self.context = context

    def retrieve_context(
        self, query_text: str, author_filter: str | None = None
    ) -> RetrievalResult:
        _ = (query_text, author_filter)
        return RetrievalResult(
            context=self.context, candidates_count=1, metadata={"provider": "stub"}
        )


def test_orchestrator_returns_provider_context():
    provider = _ProviderStub(context="provider context")
    orchestrator = AnalysisContextOrchestrator(retrieval_provider=provider)
    context = orchestrator.build_context(enhanced_query="query")
    assert context == "provider context"


def test_orchestrator_returns_empty_without_provider():
    orchestrator = AnalysisContextOrchestrator(retrieval_provider=None)
    context = orchestrator.build_context(enhanced_query="query")
    assert context == ""
