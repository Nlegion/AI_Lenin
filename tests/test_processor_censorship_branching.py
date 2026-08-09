from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

pytest.importorskip("sqlalchemy")

from src.core.processor import NewsProcessor
from src.core.safety.pre_rag_censor_types import CensorResult


@dataclass
class _News:
    id: int
    title: str
    content: str
    source: str = "TASS"


class _Repo:
    def __init__(self) -> None:
        self.marked = 0
        self.saved = 0

    async def mark_as_processed_without_analysis(self, _news_id: int) -> None:
        self.marked += 1

    async def save_analysis(self, _news_id: int, _analysis: str) -> None:
        self.saved += 1


class _CensorStub:
    def __init__(self, result: CensorResult) -> None:
        self._result = result

    async def evaluate(self, _payload):
        return self._result


class _AnalyzerStub:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_analysis(self, *_args, **_kwargs):
        self.calls += 1
        return "Краткий анализ экономики."


class _ClassifierStub:
    def should_analyze(self, *_args, **_kwargs):
        return True, "shadow"


class _ValidatorStub:
    def validate_analysis(self, *_args, **_kwargs):
        return {"is_valid": True, "score": 1.0, "reasons": []}


def _build_processor(censor_result: CensorResult) -> NewsProcessor:
    processor = NewsProcessor.__new__(NewsProcessor)
    processor.pre_rag_censor = _CensorStub(censor_result)
    processor.classifier = _ClassifierStub()
    processor.analyzer = _AnalyzerStub()
    processor.validator = _ValidatorStub()
    processor.news_guard = None
    processor.stats = {
        "news_fetched": 0,
        "news_processed": 0,
        "news_skipped": 0,
        "analyses_published": 0,
        "analyses_rejected": 0,
        "errors": 0,
    }
    return processor


def _result(decision: str) -> CensorResult:
    return CensorResult(
        decision=decision,  # type: ignore[arg-type]
        category=None,
        risk_tier="green",
        reason_codes=["test"],
        reason="test",
        message="",
        confidence={},
        context_hints=[],
        needs_yellow_warning=False,
        audit={},
        timestamp_utc=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["hard_block", "skip"])
async def test_non_allow_does_not_call_generation(decision: str) -> None:
    processor = _build_processor(_result(decision))
    repo = _Repo()
    await processor.process_single_news(_News(1, "t", "c"), repo, asyncio.Semaphore(1))
    assert repo.marked == 1
    assert repo.saved == 0
    assert processor.analyzer.calls == 0


@pytest.mark.asyncio
async def test_allow_calls_generation() -> None:
    processor = _build_processor(_result("allow"))
    repo = _Repo()
    await processor.process_single_news(_News(2, "t", "c"), repo, asyncio.Semaphore(1))
    assert processor.analyzer.calls == 1
    assert repo.saved == 1


@pytest.mark.asyncio
async def test_review_calls_generation_in_yellow_publish_mode() -> None:
    review_result = CensorResult(
        decision="review",
        category="MILITARY_OFFICIAL_STATEMENT",
        risk_tier="yellow",
        reason_codes=["test"],
        reason="test",
        message="",
        confidence={},
        context_hints=[],
        needs_yellow_warning=False,
        audit={},
        timestamp_utc=datetime.now(timezone.utc),
    )
    processor = _build_processor(review_result)
    repo = _Repo()
    await processor.process_single_news(_News(3, "t", "c"), repo, asyncio.Semaphore(1))
    assert processor.analyzer.calls == 1
    assert repo.saved == 1
