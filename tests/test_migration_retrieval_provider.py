from pathlib import Path

import pytest

from src.core.retrieval.base_provider import RetrievalResult
from src.core.retrieval.migration_provider import MigrationConfig, MigrationRetrievalProvider


class _StubProvider:
    def __init__(self, context: str, count: int, name: str):
        self.result = RetrievalResult(context=context, candidates_count=count, metadata={"provider": name})

    def retrieve_context(self, query_text: str, author_filter: str | None = None) -> RetrievalResult:
        _ = (query_text, author_filter)
        return self.result


def test_qdrant_only_mode_returns_primary():
    primary = _StubProvider(context="qdrant context", count=3, name="qdrant")
    provider = MigrationRetrievalProvider(
        primary=primary,
        shadow=None,
        config=MigrationConfig(mode="qdrant_only"),
        primary_name="qdrant",
    )
    result = provider.retrieve_context(query_text="query", author_filter="Ленин")
    assert result.context == "qdrant context"
    assert result.candidates_count == 3


def test_ab_shadow_writes_audit_and_parity_metadata(tmp_path: Path):
    primary = _StubProvider(context="капитал классовая борьба", count=2, name="qdrant")
    shadow = _StubProvider(context="капитал прибыль", count=2, name="chroma")
    audit_path = tmp_path / "audit.jsonl"
    provider = MigrationRetrievalProvider(
        primary=primary,
        shadow=shadow,
        config=MigrationConfig(mode="ab_shadow", parity_min_shared_ratio=0.1, audit_log_path=audit_path),
        primary_name="qdrant",
        shadow_name="chroma",
    )

    result = provider.retrieve_context(query_text="экономика", author_filter=None)
    assert result.metadata["mode"] == "ab_shadow"
    assert "parity_shared_ratio" in result.metadata
    assert audit_path.exists()
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"mode": "ab_shadow"' in lines[0]


def test_unsupported_mode_raises():
    primary = _StubProvider(context="x", count=1, name="qdrant")
    provider = MigrationRetrievalProvider(
        primary=primary,
        shadow=None,
        config=MigrationConfig(mode="unsupported"),
        primary_name="qdrant",
    )
    with pytest.raises(ValueError):
        provider.retrieve_context(query_text="q")
