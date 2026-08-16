import asyncio
import logging
import re
import time
from pathlib import Path
from types import SimpleNamespace

from src.core.database.repositories.news_repository import NewsRepository
from src.modules.news_system.fetcher import NewsFetcher
from src.core.lenin_analyzer import LeninAnalyzer
from src.core.publisher import PublishOutcome, TelegramPublisher
from src.core.settings.config import Settings
from src.core.utils.decorators import handle_errors
from src.core.database.db_core import session_scope
from src.core.llm.server import LeninServer
from src.modules.news_system.classifier import NewsClassifier
from src.core.analysis_validator import AnalysisValidator
from src.core.generation.postprocess_clean import scrub_after_output_guard
from src.core.news_item_pipeline import (
    attach_censor_cache_callbacks,
    evaluate_and_annotate_news,
    generate_and_persist_analysis,
)
from src.core.ops.report_formatter import format_ops_digest
from src.core.ops.window_stats import WindowStats
from src.core.safety.news_guard import NewsGuard
from src.core.safety.safety_gate import SafetyGate
from src.core.safety.pre_rag_censor import PreRagCensor
from src.core.settings.censorship_runtime_config import (
    default_censorship_runtime_config_path,
    load_censorship_runtime_config,
)
from src.core.settings.generation_config import (
    llm_spawn_local_from_env,
    provider_from_env,
)
from src.core.settings.ops_report_config import (
    default_ops_report_path,
    load_ops_report_config,
)

logger = logging.getLogger(__name__)
db_lock = asyncio.Lock()
_WS_RE = re.compile(r"\s+")


class _LockScopedRepo:
    """Open a short DB session under db_lock for each write (LLM stays unlocked)."""

    def __init__(self, lock: asyncio.Lock) -> None:
        self._lock = lock

    async def mark_as_processed_without_analysis(self, news_id: str) -> None:
        async with self._lock:
            async with session_scope() as session:
                await NewsRepository(session).mark_as_processed_without_analysis(
                    news_id
                )

    async def save_analysis(self, news_id: str, analysis: str) -> None:
        async with self._lock:
            async with session_scope() as session:
                await NewsRepository(session).save_analysis(news_id, analysis)


class NewsProcessor:
    def __init__(self):
        self.config = Settings()
        logger.info("Инициализация EnhancedNewsProcessor с раздельными циклами")

        self.fetcher = NewsFetcher()
        self.analyzer = None
        self.spawn_local = llm_spawn_local_from_env()
        self.server = LeninServer() if self.spawn_local else None
        self.classifier = NewsClassifier()
        self.validator = AnalysisValidator()
        self.news_guard = self._init_news_guard()
        self.safety_gate = self._init_safety_gate()
        self.pre_rag_censor = self._init_pre_rag_censor()
        self.analyzer_ready = asyncio.Event()
        self.ops_config = load_ops_report_config(
            path=default_ops_report_path(Path(self.config.BASE_DIR))
        )
        try:
            llm_provider = provider_from_env()
        except ValueError:
            llm_provider = "unknown"

        asyncio.create_task(self.initialize_components())

        self.publisher = TelegramPublisher()
        self.last_fetch_time = 0
        self.fetch_interval = 300
        self.stats: WindowStats = WindowStats(
            max_latency_samples=self.ops_config.max_latency_samples,
            llm_provider=str(llm_provider),
        )
        self.analysis_feedback = {}

    def _init_news_guard(self):
        config_path = Path(self.config.BASE_DIR) / "config" / "news_guard.yaml"
        if not config_path.exists():
            logger.warning("NewsGuard config not found: %s", config_path)
            return None
        try:
            return NewsGuard.from_file(path=config_path)
        except Exception as error:  # noqa: BLE001
            logger.exception("Failed to initialize NewsGuard: %s", error)
            return None

    def _init_safety_gate(self) -> SafetyGate | None:
        try:
            return SafetyGate.from_base_dir(Path(self.config.BASE_DIR))
        except Exception as error:  # noqa: BLE001
            logger.exception("Failed to initialize SafetyGate: %s", error)
            return None

    def _init_pre_rag_censor(self) -> PreRagCensor | None:
        try:
            cfg_path = default_censorship_runtime_config_path(
                Path(self.config.BASE_DIR)
            )
            runtime_cfg = load_censorship_runtime_config(cfg_path)
            return PreRagCensor(
                safety_gate=self.safety_gate,
                news_guard=self.news_guard,
                config=runtime_cfg,
                config_path=str(cfg_path),
            )
        except Exception as error:  # noqa: BLE001
            logger.exception("Failed to initialize pre-RAG censor: %s", error)
            return None

    def _deduplicate_news_batch(self, news_items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        deduped: list[dict] = []
        for item in news_items:
            title = str(item.get("title") or "")
            body = str(item.get("content") or "")
            key = _WS_RE.sub(" ", f"{title}\n{body}".strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        dropped = len(news_items) - len(deduped)
        if dropped > 0:
            logger.info(
                "Pre-censor batch dedup dropped=%s kept=%s", dropped, len(deduped)
            )
        return deduped

    async def _backlog_counts(self) -> tuple[int, int, int]:
        async with db_lock:
            async with session_scope() as session:
                repo = NewsRepository(session)
                workable = await repo.count_unprocessed(hours=24)
                stale = await repo.count_stale_unprocessed(hours=24)
                unpublished = await repo.count_unpublished()
        return workable, stale, unpublished

    async def initialize_components(self):
        """Initialize components; one compact boot Telegram on success."""
        try:
            logger.info("Начало инициализации системы ИИ-Ленин")

            if self.spawn_local:
                logger.info("Запуск сервера llama.cpp...")
                if self.server is None or not await self.server.start_server():
                    error_msg = "Не удалось запустить сервер llama.cpp"
                    await self.publisher.send_admin_notification(f"❌ {error_msg}")
                    raise Exception(error_msg)
                logger.info("Сервер llama.cpp запущен")
                mode = "local llama"
            else:
                logger.info("LLM_SPAWN_LOCAL=false; skipping local llama-server")
                mode = f"remote {self.stats.llm_provider}"

            logger.info("Инициализация анализатора...")
            self.analyzer = LeninAnalyzer()
            await self.analyzer.initialize_session()
            logger.info("Анализатор инициализирован")

            self.analyzer_ready.set()
            workable, _stale, unpublished = await self._backlog_counts()
            await self.publisher.send_admin_notification(
                f"Запуск OK | {mode} | backlog workable {workable} / unpublished {unpublished}"
            )
            logger.info("Все компоненты инициализированы")

        except Exception as e:
            logger.exception(f"Ошибка инициализации: {str(e)}")
            self.analyzer_ready.set()
            await self.publisher.send_admin_notification(
                f"❌ Критическая ошибка инициализации: {str(e)[:300]}"
            )

    @handle_errors
    async def fetch_news_cycle(self):
        """Цикл сбора новостей (запускается каждые 5 минут)"""
        while True:
            try:
                current_time = time.time()

                if current_time - self.last_fetch_time >= self.fetch_interval:
                    news_items = self.fetcher.fetch_all()
                    rss_seen = len(news_items)
                    deduped = self._deduplicate_news_batch(news_items)
                    dedup_dropped = rss_seen - len(deduped)

                    async with db_lock:
                        async with session_scope() as session:
                            repo = NewsRepository(session)
                            inserted = await repo.save_news(deduped)

                    self.stats["rss_seen"] += rss_seen
                    self.stats["dedup_dropped"] += dedup_dropped
                    self.stats["inserted"] += inserted
                    self.last_fetch_time = current_time

                    logger.info(
                        "Сбор новостей: rss=%s dedup_dropped=%s inserted=%s",
                        rss_seen,
                        dedup_dropped,
                        inserted,
                    )
                    if (
                        inserted > 0
                        and self.ops_config.fetch_notify == "new_only"
                    ):
                        db_dupes = max(0, len(deduped) - inserted)
                        await self.publisher.send_admin_notification(
                            f"Новых {inserted} из TASS "
                            f"(RSS {rss_seen}, дубликаты {db_dupes + dedup_dropped})"
                        )

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Ошибка в цикле сбора новостей: {str(e)}")
                await self.publisher.send_admin_notification(
                    f"❌ Ошибка сбора новостей: {str(e)[:200]}"
                )
                await asyncio.sleep(60)

    @handle_errors
    async def process_news_cycle(self):
        """Обработка ожидающих новостей с улучшенной фильтрацией"""
        if not self.analyzer_ready.is_set():
            logger.info("Ожидание инициализации анализатора...")
            await self.analyzer_ready.wait()

        processing_semaphore = asyncio.Semaphore(2)

        while True:
            try:
                async with db_lock:
                    async with session_scope() as session:
                        repo = NewsRepository(session)
                        unprocessed = await repo.get_unprocessed_news(limit=5)
                        news_ids = [news.id for news in unprocessed]
                        logger.info("Найдено %s необработанных новостей", len(news_ids))

                processing_tasks = []
                for news_id in news_ids:
                    task = asyncio.create_task(
                        self.process_single_news_by_id(news_id, processing_semaphore)
                    )
                    processing_tasks.append(task)
                if processing_tasks:
                    await asyncio.gather(*processing_tasks)

                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Ошибка в цикле обработки новостей: {str(e)}")
                await self.publisher.send_admin_notification(
                    f"❌ Ошибка обработки новостей: {str(e)[:200]}"
                )
                await asyncio.sleep(30)

    async def process_single_news_by_id(
        self, news_id: str | int, semaphore: asyncio.Semaphore
    ) -> None:
        """Load by PK under lock; run censor under lock; generate outside lock."""
        async with semaphore:
            snap: SimpleNamespace | None = None
            annotate = None
            async with db_lock:
                async with session_scope() as session:
                    repo = NewsRepository(session)
                    news = await repo.get_news_by_id(str(news_id))
                    if news is None:
                        logger.warning(
                            "news_id=%s disappeared before processing", news_id
                        )
                        self.stats["errors"] += 1
                        return
                    snap = SimpleNamespace(
                        id=news.id,
                        title=news.title,
                        content=news.content,
                        source=news.source,
                        url=news.url,
                    )
                    if self.pre_rag_censor is None:
                        logger.error(
                            "PreRagCensor unavailable; holding news_id=%s",
                            news.id,
                        )
                        await repo.mark_as_processed_without_analysis(news.id)
                        self.stats["news_skipped"] += 1
                        return
                    attach_censor_cache_callbacks(
                        censor=self.pre_rag_censor, repo=repo
                    )
                    annotate = await evaluate_and_annotate_news(
                        news=snap,
                        censor=self.pre_rag_censor,
                        classifier=self.classifier,
                        base_dir=Path(self.config.BASE_DIR),
                    )
                    if annotate.stop == "skip":
                        await repo.mark_as_processed_without_analysis(snap.id)
                        self.stats["news_skipped"] += 1
                        self.stats.record_skip_reasons(annotate.reason_codes)
                        return
                    if annotate.decision == "review":
                        self.stats["review_continued"] += 1

            try:
                await generate_and_persist_analysis(
                    news=snap,
                    analyzer=self.analyzer,
                    news_guard=self.news_guard,
                    validator=self.validator,
                    repo=_LockScopedRepo(db_lock),
                    stats=self.stats,
                )
            except Exception as e:
                logger.error("Ошибка обработки новости %s: %s", news_id, e)
                self.stats["errors"] += 1

    async def _process_loaded_news(self, *, news, repo) -> None:
        """Test-friendly path: caller owns session/lock."""
        logger.info("Обработка новости: %s...", news.title[:50])
        if self.pre_rag_censor is None:
            logger.error(
                "PreRagCensor unavailable; holding news_id=%s for review", news.id
            )
            await repo.mark_as_processed_without_analysis(news.id)
            self.stats["news_skipped"] += 1
            return
        try:
            attach_censor_cache_callbacks(censor=self.pre_rag_censor, repo=repo)
            annotate = await evaluate_and_annotate_news(
                news=news,
                censor=self.pre_rag_censor,
                classifier=self.classifier,
                base_dir=Path(self.config.BASE_DIR),
            )
            if annotate.stop == "skip":
                await repo.mark_as_processed_without_analysis(news.id)
                self.stats["news_skipped"] += 1
                if isinstance(self.stats, WindowStats):
                    self.stats.record_skip_reasons(annotate.reason_codes)
                return
            if annotate.decision == "review":
                self.stats["review_continued"] = (
                    self.stats.get("review_continued", 0) + 1
                )
            await generate_and_persist_analysis(
                news=news,
                analyzer=self.analyzer,
                news_guard=self.news_guard,
                validator=self.validator,
                repo=repo,
                stats=self.stats,
            )
        except Exception as e:
            logger.error("Ошибка обработки новости %s: %s", news.id, e)
            self.stats["errors"] += 1

    async def process_single_news(self, news, repo, semaphore):
        """Backward-compatible wrapper used by older tests."""
        async with semaphore:
            await self._process_loaded_news(news=news, repo=repo)

    @handle_errors
    async def publish_cycle(self):
        """Publish cycle: short DB critical sections; sleep outside lock."""
        while True:
            try:
                unpublished = []
                async with db_lock:
                    async with session_scope() as session:
                        repo = NewsRepository(session)
                        rows = await repo.get_unpublished_analysis(limit=5)
                        for item in rows:
                            unpublished.append(
                                SimpleNamespace(
                                    news_id=item.news_id,
                                    analysis=item.analysis,
                                    title=item.news.title,
                                    url=item.news.url,
                                )
                            )

                if unpublished:
                    logger.info(
                        "Найдено %s анализов для публикации", len(unpublished)
                    )

                for item in unpublished:
                    validation = self.validator.validate_analysis(
                        item.analysis, item.title
                    )
                    if not validation["is_valid"]:
                        logger.warning(
                            "Анализ %s отклонен: %s",
                            item.news_id,
                            ", ".join(validation["reasons"]),
                        )
                        async with db_lock:
                            async with session_scope() as session:
                                await NewsRepository(session).mark_analysis_discarded(
                                    item.news_id
                                )
                        self.stats["analyses_rejected"] += 1
                        if isinstance(self.stats, WindowStats):
                            self.stats.record_reject_reasons(
                                list(validation.get("reasons") or [])
                            )
                        continue

                    try:
                        analysis_to_publish = item.analysis
                        if self.news_guard is not None:
                            guard_result = self.news_guard.guard_output(
                                analysis=item.analysis
                            )
                            analysis_to_publish = scrub_after_output_guard(
                                guard_result.moderated_text
                            )
                        outcome = await self.publisher.publish_analysis(
                            item.news_id,
                            item.title,
                            item.url,
                            analysis_to_publish,
                        )
                        if outcome == PublishOutcome.SUCCESS:
                            async with db_lock:
                                async with session_scope() as session:
                                    await NewsRepository(session).mark_as_published(
                                        item.news_id
                                    )
                            self.stats["analyses_published"] += 1
                            logger.info(
                                "Анализ %s успешно опубликован", item.news_id
                            )
                            await asyncio.sleep(30)
                        elif outcome == PublishOutcome.PERMANENT_REJECT:
                            async with db_lock:
                                async with session_scope() as session:
                                    await NewsRepository(
                                        session
                                    ).mark_analysis_discarded(item.news_id)
                            self.stats["analyses_rejected"] += 1
                            if isinstance(self.stats, WindowStats):
                                self.stats.record_reject_reasons(["triad_missing"])
                            logger.warning(
                                "Анализ %s отброшен (permanent)", item.news_id
                            )
                        else:
                            self.stats["publish_failed"] += 1
                            logger.warning(
                                "Неудачная попытка публикации анализа %s",
                                item.news_id,
                            )
                    except Exception as e:
                        self.stats["publish_failed"] += 1
                        logger.error(
                            "Ошибка публикации анализа %s: %s",
                            item.news_id,
                            str(e),
                        )

                await asyncio.sleep(15)

            except Exception as e:
                logger.error(f"Ошибка в цикле публикации: {str(e)}")
                await asyncio.sleep(30)

    async def report_cycle(self):
        """Sleep first, then send a real window digest."""
        interval = max(60, int(self.ops_config.interval_seconds))
        while True:
            try:
                await asyncio.sleep(interval)
                workable, stale, unpublished = await self._backlog_counts()
                snapshot = self.stats.snapshot_and_reset()
                report = format_ops_digest(
                    snapshot,
                    interval_seconds=interval,
                    workable_backlog=workable,
                    stale_backlog=stale,
                    unpublished=unpublished,
                    top_reasons=self.ops_config.top_reasons,
                    idle_digest=self.ops_config.idle_digest,
                )
                logger.info(report)
                await self.publisher.send_admin_notification(report)
            except Exception as e:
                logger.error(f"Ошибка в цикле отчетности: {str(e)}")
                await asyncio.sleep(interval)

    async def start_separated_processing(self):
        """Запуск раздельных циклов обработки"""
        logger.info("Запуск раздельных циклов обработки")
        await asyncio.gather(
            self.fetch_news_cycle(),
            self.process_news_cycle(),
            self.publish_cycle(),
            self.report_cycle(),
        )

    @handle_errors
    async def close(self):
        """Закрытие ресурсов"""
        await self.publisher.send_admin_notification(
            "🛑 Завершение работы системы ИИ-Ленин"
        )

        if self.analyzer:
            await self.analyzer.close_session()
        if self.server is not None:
            await self.server.stop_server()

        await self.publisher.send_admin_notification("✅ Система успешно остановлена")
