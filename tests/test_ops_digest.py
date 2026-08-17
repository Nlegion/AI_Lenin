"""Unit tests for admin ops digest (WindowStats, formatter, funnel hooks)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.database.utc import utc_now
from src.core.ops.report_formatter import format_ops_digest
from src.core.ops.window_stats import WindowStats
from src.core.publisher import PublishOutcome, TelegramPublisher
from src.core.settings.ops_report_config import OpsReportConfig, load_ops_report_config


def test_window_stats_dict_like_and_alias() -> None:
    stats = WindowStats()
    stats["news_processed"] += 1
    stats["news_fetched"] = 5
    assert stats["rss_seen"] == 5
    assert "news_fetched" in stats
    assert stats.get("missing", 9) == 9
    snap = stats.snapshot_and_reset()
    assert snap.counters["rss_seen"] == 5
    assert stats["rss_seen"] == 0


def test_window_stats_circuit_deltas_not_overwrite() -> None:
    stats = WindowStats()
    stats.sync_circuit_deltas(total_timeouts=3, total_opens=1)
    assert stats["generation_timeouts"] == 3
    assert stats["circuit_opens"] == 1
    stats.sync_circuit_deltas(total_timeouts=5, total_opens=1)
    assert stats["generation_timeouts"] == 5
    assert stats["circuit_opens"] == 1
    snap = stats.snapshot_and_reset()
    assert snap.counters["generation_timeouts"] == 5
    stats.sync_circuit_deltas(total_timeouts=5, total_opens=2)
    assert stats["generation_timeouts"] == 0
    assert stats["circuit_opens"] == 1


def test_window_stats_latency_skips_none_and_caps() -> None:
    stats = WindowStats(max_latency_samples=3)
    stats.record_latency_ms(None)
    stats.record_latency_ms(100)
    stats.record_latency_ms(200)
    stats.record_latency_ms(300)
    stats.record_latency_ms(400)
    assert stats.latency_samples_ms == [200, 300, 400]
    snap = stats.snapshot()
    assert snap.percentile(50) == 300


def test_format_ops_digest_idle_short() -> None:
    stats = WindowStats(llm_provider="deepseek")
    stats["rss_seen"] = 100
    snap = stats.snapshot()
    text = format_ops_digest(
        snap,
        interval_seconds=1800,
        workable_backlog=0,
        stale_backlog=0,
        unpublished=0,
        idle_digest="short",
    )
    assert "Тихо" in text
    assert "deepseek" in text
    assert "Воронка" not in text


def test_format_ops_digest_stall_with_backlog() -> None:
    stats = WindowStats(llm_provider="deepseek")
    snap = stats.snapshot()
    text = format_ops_digest(
        snap,
        interval_seconds=1800,
        workable_backlog=2,
        stale_backlog=1,
        unpublished=0,
        idle_digest="short",
    )
    assert "простой" in text
    assert "рабочих 2" in text


def test_format_ops_digest_busy_funnel_and_latency() -> None:
    stats = WindowStats(llm_provider="deepseek")
    stats["rss_seen"] = 100
    stats["dedup_dropped"] = 2
    stats["inserted"] = 3
    stats["news_processed"] = 2
    stats["analyses_published"] = 1
    stats["news_skipped"] = 1
    stats.record_skip_reasons(["combat"])
    stats.record_latency_ms(8100)
    stats.record_latency_ms(14000)
    snap = stats.snapshot()
    text = format_ops_digest(snap, interval_seconds=1800, workable_backlog=1)
    assert "Воронка: RSS 100 / дедуп 2 / новых 3" in text
    assert "combat:1" in text
    assert "p50" in text


def test_ops_report_config_defaults(tmp_path) -> None:
    missing = tmp_path / "missing.yaml"
    cfg = load_ops_report_config(missing)
    assert cfg.interval_seconds == 21600
    assert cfg.fetch_notify == "never"


def test_ops_report_config_yaml(tmp_path) -> None:
    path = tmp_path / "ops_report.yaml"
    path.write_text(
        "ops_report:\n  interval_seconds: 60\n  fetch_notify: never\n",
        encoding="utf-8",
    )
    cfg = load_ops_report_config(path)
    assert cfg.interval_seconds == 60
    assert cfg.fetch_notify == "never"
    assert isinstance(cfg, OpsReportConfig)


@pytest.mark.asyncio
async def test_publish_outcome_triad_missing() -> None:
    publisher = TelegramPublisher()
    publisher.service = MagicMock()
    publisher.service.send_message = AsyncMock(return_value={"ok": True})
    outcome = await publisher.publish_analysis(
        news_id="1",
        title="t",
        url="https://example.com",
        analysis="нет секций",
    )
    assert outcome == PublishOutcome.PERMANENT_REJECT
    publisher.service.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_publish_outcome_transient_api_fail() -> None:
    publisher = TelegramPublisher()
    publisher.service = MagicMock()
    publisher.service.send_message = AsyncMock(return_value={"ok": False})
    outcome = await publisher.publish_analysis(
        news_id="1",
        title="t",
        url="https://example.com",
        analysis="Механизм: a.\n\nВывод: b.",
    )
    assert outcome == PublishOutcome.TRANSIENT_FAIL


@pytest.mark.asyncio
async def test_publish_outcome_success() -> None:
    publisher = TelegramPublisher()
    publisher.service = MagicMock()
    publisher.service.send_message = AsyncMock(return_value={"ok": True})
    outcome = await publisher.publish_analysis(
        news_id="1",
        title="t",
        url="https://example.com",
        analysis="Механизм: a.\n\nВывод: b.",
    )
    assert outcome == PublishOutcome.SUCCESS


@pytest.mark.asyncio
async def test_report_cycle_sleeps_before_send() -> None:
    from src.core.processor import NewsProcessor

    processor = NewsProcessor.__new__(NewsProcessor)
    processor.ops_config = OpsReportConfig(interval_seconds=60)
    processor.stats = WindowStats(llm_provider="deepseek")
    processor.publisher = MagicMock()
    processor.publisher.send_admin_notification = AsyncMock(return_value=True)
    processor._backlog_counts = AsyncMock(return_value=(0, 0, 0))

    sleeps: list[float] = []

    async def _sleep(seconds: float):
        sleeps.append(seconds)
        if len(sleeps) >= 1 and processor.publisher.send_admin_notification.await_count:
            raise asyncio.CancelledError()
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    with patch("src.core.processor.asyncio.sleep", side_effect=_sleep):
        with pytest.raises(asyncio.CancelledError):
            await processor.report_cycle()

    assert sleeps[0] == 60
    assert processor.publisher.send_admin_notification.await_count == 1
    sent = processor.publisher.send_admin_notification.await_args.args[0]
    assert "Тихо" in sent or "Воронка" in sent or "мин" in sent


@pytest.mark.asyncio
async def test_start_separated_processing_no_admin_ping() -> None:
    from src.core.processor import NewsProcessor

    processor = NewsProcessor.__new__(NewsProcessor)
    processor.publisher = MagicMock()
    processor.publisher.send_admin_notification = AsyncMock(return_value=True)

    async def _noop():
        return None

    processor.fetch_news_cycle = _noop  # type: ignore[method-assign]
    processor.process_news_cycle = _noop  # type: ignore[method-assign]
    processor.publish_cycle = _noop  # type: ignore[method-assign]
    processor.report_cycle = _noop  # type: ignore[method-assign]

    await processor.start_separated_processing()
    processor.publisher.send_admin_notification.assert_not_called()


@pytest.mark.asyncio
async def test_process_by_id_missing_increments_errors() -> None:
    from src.core.processor import NewsProcessor

    processor = NewsProcessor.__new__(NewsProcessor)
    processor.stats = WindowStats()
    processor.pre_rag_censor = object()
    processor.classifier = MagicMock()
    processor.analyzer = MagicMock()
    processor.news_guard = None
    processor.validator = MagicMock()
    processor.config = SimpleNamespace(BASE_DIR=".")

    fake_repo = MagicMock()
    fake_repo.get_news_by_id = AsyncMock(return_value=None)

    class _Scope:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, *args):
            return False

    with patch("src.core.processor.session_scope", return_value=_Scope()):
        with patch("src.core.processor.NewsRepository", return_value=fake_repo):
            await processor.process_single_news_by_id("missing", asyncio.Semaphore(1))

    assert processor.stats["errors"] == 1


@pytest.mark.asyncio
async def test_pipeline_records_latency_not_cache_hit() -> None:
    from src.core.news_item_pipeline import generate_and_persist_analysis

    stats = WindowStats()

    class _Analyzer:
        last_pipeline_metadata = {"latency_ms": 1234, "cache_hit": False}

        async def generate_analysis(self, *args, **kwargs):
            return (
                "Механизм: капиталистическая конкуренция.\n\n"
                "Вывод: трудящиеся платят цену."
            )

    class _Validator:
        def validate_analysis(self, *args, **kwargs):
            return {"is_valid": True, "score": 1.0, "reasons": []}

    class _Repo:
        async def save_analysis(self, *args, **kwargs):
            return None

    news = SimpleNamespace(
        id="1", title="t", content="c", _risk_tier="green", _context_hints=[]
    )
    with patch(
        "src.core.news_item_pipeline.is_publishable_analysis", return_value=True
    ):
        await generate_and_persist_analysis(
            news=news,
            analyzer=_Analyzer(),
            news_guard=None,
            validator=_Validator(),
            repo=_Repo(),
            stats=stats,
        )
    assert stats.latency_samples_ms == [1234]
    assert stats["news_processed"] == 1


@pytest.mark.asyncio
async def test_pipeline_skips_latency_on_cache_hit() -> None:
    from src.core.news_item_pipeline import generate_and_persist_analysis

    stats = WindowStats()
    stats.record_latency_ms(999)

    class _Analyzer:
        last_pipeline_metadata = {"latency_ms": 50, "cache_hit": True}

        async def generate_analysis(self, *args, **kwargs):
            return "отказываюсь от анализа данной темы"

    class _Repo:
        async def mark_as_processed_without_analysis(self, *args, **kwargs):
            return None

    news = SimpleNamespace(id="1", title="t", content="c")
    await generate_and_persist_analysis(
        news=news,
        analyzer=_Analyzer(),
        news_guard=None,
        validator=MagicMock(),
        repo=_Repo(),
        stats=stats,
    )
    assert stats.latency_samples_ms == [999]
    assert stats["news_skipped"] == 1


@pytest.mark.asyncio
async def test_save_news_returns_inserted_not_conflicts() -> None:
    pytest.importorskip("aiosqlite")
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from src.core.database.models.models import Base
    from src.core.database.repositories.news_repository import NewsRepository

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    now = utc_now()
    item = {
        "id": "n1",
        "title": "t",
        "content": "c",
        "source": "TASS",
        "date": now,
        "url": "https://example.com/1",
    }
    async with Session() as session:
        repo = NewsRepository(session)
        first = await repo.save_news([item])
        await session.commit()
        assert first == 1
        second = await repo.save_news([item])
        await session.commit()
        assert second == 0

    await engine.dispose()


@pytest.mark.asyncio
async def test_mark_analysis_discarded_leaves_unpublished_queue() -> None:
    pytest.importorskip("aiosqlite")
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from src.core.database.models.models import Analysis, Base, News
    from src.core.database.repositories.news_repository import NewsRepository

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    now = utc_now()
    async with Session() as session:
        session.add(
            News(
                id="n1",
                title="t",
                content="c",
                source="TASS",
                date=now,
                url="https://example.com/1",
                processed=True,
                processed_at=now,
                created_at=now,
            )
        )
        session.add(
            Analysis(
                news_id="n1",
                analysis="bad",
                published=False,
                published_at=None,
                created_at=now,
            )
        )
        await session.commit()

        repo = NewsRepository(session)
        assert await repo.count_unpublished() == 1
        await repo.mark_analysis_discarded("n1")
        await session.commit()
        assert await repo.count_unpublished() == 0

    await engine.dispose()


@pytest.mark.asyncio
async def test_pipeline_output_guard_blocked_bucket() -> None:
    from src.core.news_item_pipeline import generate_and_persist_analysis

    stats = WindowStats()

    class _Analyzer:
        last_pipeline_metadata = {"latency_ms": 10}

        async def generate_analysis(self, *args, **kwargs):
            return "Механизм: x.\n\nВывод: y."

    class _Guard:
        def guard_output(self, **kwargs):
            return SimpleNamespace(
                blocked=True,
                moderated_text="safe",
                reason_codes=["block:x"],
            )

    class _Validator:
        def validate_analysis(self, *args, **kwargs):
            return {"is_valid": False, "score": 0.0, "reasons": ["too_short"]}

    class _Repo:
        async def mark_as_processed_without_analysis(self, *args, **kwargs):
            return None

    news = SimpleNamespace(
        id="1", title="t", content="c", _risk_tier="green", _context_hints=[]
    )
    await generate_and_persist_analysis(
        news=news,
        analyzer=_Analyzer(),
        news_guard=_Guard(),
        validator=_Validator(),
        repo=_Repo(),
        stats=stats,
    )
    assert stats["output_guard_blocked"] == 1
    assert stats["analyses_rejected"] == 1

    from src.core.processor import NewsProcessor
    from src.core.safety.pre_rag_censor_types import CensorResult

    processor = NewsProcessor.__new__(NewsProcessor)
    processor.config = SimpleNamespace(BASE_DIR=".")
    processor.stats = WindowStats()
    processor.news_guard = None
    processor.validator = MagicMock(
        validate_analysis=MagicMock(
            return_value={"is_valid": True, "score": 1.0, "reasons": []}
        )
    )

    class _Analyzer:
        calls = 0
        last_pipeline_metadata = {"latency_ms": 10}

        async def generate_analysis(self, *args, **kwargs):
            self.calls += 1
            return "Механизм: x.\n\nВывод: y."

    processor.analyzer = _Analyzer()
    processor.classifier = MagicMock(
        should_analyze=MagicMock(return_value=(True, "ok"))
    )

    result = CensorResult(
        decision="review",
        category="MILITARY_OFFICIAL_STATEMENT",
        risk_tier="yellow",
        reason_codes=["yellow_military"],
        reason="test",
        message="",
        confidence={},
        context_hints=[],
        needs_yellow_warning=False,
        audit={},
        timestamp_utc=datetime.now(UTC),
    )

    class _Censor:
        async def evaluate(self, *_a, **_k):
            return result

    processor.pre_rag_censor = _Censor()

    class _Repo:
        saved = 0

        async def mark_as_processed_without_analysis(self, *_a, **_k):
            return None

        async def save_analysis(self, *_a, **_k):
            self.saved += 1

    repo = _Repo()
    news = SimpleNamespace(id="3", title="t", content="c", source="TASS", url="u")
    with (
        patch("src.core.news_item_pipeline.append_yellow_audit"),
        patch("src.core.news_item_pipeline.is_publishable_analysis", return_value=True),
    ):
        await processor.process_single_news(news, repo, asyncio.Semaphore(1))
    assert processor.stats["review_continued"] == 1
    assert processor.analyzer.calls == 1
    assert repo.saved == 1
