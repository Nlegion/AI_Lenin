import asyncio
import time
import logging
import re
from pathlib import Path
from src.core.database.repositories.news_repository import NewsRepository
from src.modules.news_system.fetcher import NewsFetcher
from src.core.lenin_analyzer import LeninAnalyzer
from src.core.publisher import TelegramPublisher
from src.core.settings.config import Settings
from src.core.utils.decorators import handle_errors
from src.core.database.db_core import session_scope
from src.core.llama_server import LeninServer
from src.modules.news_system.classifier import NewsClassifier
from src.core.analysis_validator import AnalysisValidator
from src.core.generation.postprocess_clean import scrub_after_output_guard
from src.core.news_item_pipeline import (
    attach_censor_cache_callbacks,
    evaluate_and_annotate_news,
    generate_and_persist_analysis,
)
from src.core.safety.news_guard import NewsGuard
from src.core.safety.safety_gate import SafetyGate
from src.core.safety.pre_rag_censor import PreRagCensor
from src.core.settings.censorship_runtime_config import (
    default_censorship_runtime_config_path,
    load_censorship_runtime_config,
)

logger = logging.getLogger(__name__)
db_lock = asyncio.Lock()
_WS_RE = re.compile(r"\s+")

class NewsProcessor:
    def __init__(self):
        self.config = Settings()
        logger.info("Инициализация EnhancedNewsProcessor с раздельными циклами")

        self.fetcher = NewsFetcher()
        self.analyzer = None
        self.server = LeninServer()
        self.classifier = NewsClassifier()
        self.validator = AnalysisValidator()
        self.news_guard = self._init_news_guard()
        self.safety_gate = self._init_safety_gate()
        self.pre_rag_censor = self._init_pre_rag_censor()
        self.analyzer_ready = asyncio.Event()

        asyncio.create_task(self.initialize_components())

        self.publisher = TelegramPublisher()

        # Время последнего сбора новостей
        self.last_fetch_time = 0
        self.fetch_interval = 300  # 5 минут между сборами новостей

        # Статистика
        self.stats = {
            "news_fetched": 0,
            "news_processed": 0,
            "news_skipped": 0,
            "analyses_published": 0,
            "analyses_rejected": 0,
            "errors": 0,
            "generation_timeouts": 0,
            "circuit_opens": 0,
            "degraded_held": 0,
        }

        # Словарь для хранения замечаний к анализам
        self.analysis_feedback = {}
        self.failed_attempts = {}

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
            cfg_path = default_censorship_runtime_config_path(Path(self.config.BASE_DIR))
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
            logger.info("Pre-censor batch dedup dropped=%s kept=%s", dropped, len(deduped))
        return deduped

    async def initialize_components(self):
        """Параллельная инициализация компонентов с запуском сервера"""
        try:
            # Отправляем уведомление о начале инициализации
            await self.publisher.send_admin_notification("🔄 Начало инициализации системы ИИ-Ленин")

            # Запускаем сервер llama.cpp
            logger.info("Запуск сервера llama.cpp...")
            await self.publisher.send_admin_notification("🔌 Запуск сервера llama.cpp...")

            if not await self.server.start_server():
                error_msg = "Не удалось запустить сервер llama.cpp"
                await self.publisher.send_admin_notification(f"❌ {error_msg}")
                raise Exception(error_msg)

            await self.publisher.send_admin_notification("✅ Сервер llama.cpp запущен")

            # Default persona_model=base_strong from config/generation.yaml (fine_tuned optional).
            await self.publisher.send_admin_notification("🧠 Инициализация анализатора...")
            self.analyzer = LeninAnalyzer()
            await self.analyzer.initialize_session()
            await self.publisher.send_admin_notification("✅ Анализатор инициализирован")

            self.analyzer_ready.set()
            logger.info("Все компоненты инициализированы")

            # Отправляем уведомление о успешном запуске
            await self.publisher.send_admin_notification("🚀 Система ИИ-Ленин успешно запущена и готова к работе!")

        except Exception as e:
            logger.exception(f"Ошибка инициализации: {str(e)}")
            # Устанавливаем флаг готовности даже при ошибке, чтобы циклы не зависали
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

                # Проверяем, прошло ли достаточно времени с последнего сбора
                if current_time - self.last_fetch_time >= self.fetch_interval:
                    await self.publisher.send_admin_notification("📡 Начало сбора новостей...")

                    news_items = self.fetcher.fetch_all()
                    news_items = self._deduplicate_news_batch(news_items)

                    if news_items:
                        await self.publisher.send_admin_notification(
                            f"📊 Собрано {len(news_items)} новостей из TASS"
                        )

                    # Используем блокировку для доступа к базе данных
                    async with db_lock:
                        async with session_scope() as session:
                            repo = NewsRepository(session)
                            await repo.save_news(news_items)

                    self.stats["news_fetched"] += len(news_items)
                    self.last_fetch_time = current_time

                    logger.info(f"Сбор новостей завершен. Всего собрано: {self.stats['news_fetched']}")

                # Ждем 1 минуту перед следующей проверкой
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Ошибка в цикле сбора новостей: {str(e)}")
                await self.publisher.send_admin_notification(f"❌ Ошибка сбора новостей: {str(e)[:200]}")
                await asyncio.sleep(60)  # Ждем перед повторной попыткой

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
                    if news_id in self.failed_attempts:
                        del self.failed_attempts[news_id]
                    task = asyncio.create_task(
                        self.process_single_news_by_id(news_id, processing_semaphore)
                    )
                    processing_tasks.append(task)
                if processing_tasks:
                    await asyncio.gather(*processing_tasks)

                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Ошибка в цикле обработки новостей: {str(e)}")
                await self.publisher.send_admin_notification(f"❌ Ошибка обработки новостей: {str(e)[:200]}")
                await asyncio.sleep(30)

    async def process_single_news_by_id(self, news_id: int, semaphore: asyncio.Semaphore) -> None:
        """Process one news item with an isolated DB session/repository."""
        async with semaphore:
            async with db_lock:
                async with session_scope() as session:
                    repo = NewsRepository(session)
                    pending = await repo.get_unprocessed_news(limit=20)
                    news = next((item for item in pending if item.id == news_id), None)
                    if news is None:
                        logger.warning("news_id=%s disappeared before processing", news_id)
                        return
                    await self._process_loaded_news(news=news, repo=repo)

    async def _process_loaded_news(self, *, news, repo) -> None:
        logger.info("Обработка новости: %s...", news.title[:50])
        if self.pre_rag_censor is None:
            logger.error("PreRagCensor unavailable; holding news_id=%s for review", news.id)
            await repo.mark_as_processed_without_analysis(news.id)
            self.stats["news_skipped"] += 1
            return
        try:
            attach_censor_cache_callbacks(censor=self.pre_rag_censor, repo=repo)
            stop = await evaluate_and_annotate_news(
                news=news,
                censor=self.pre_rag_censor,
                classifier=self.classifier,
                base_dir=Path(self.config.BASE_DIR),
            )
            if stop == "skip":
                await repo.mark_as_processed_without_analysis(news.id)
                self.stats["news_skipped"] += 1
                return
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
        """Упрощенный цикл публикации анализов с задержкой"""
        while True:
            try:
                async with db_lock:
                    async with session_scope() as session:
                        repo = NewsRepository(session)
                        unpublished = await repo.get_unpublished_analysis(limit=5)  # Уменьшили лимит

                        if unpublished:
                            logger.info(f"Найдено {len(unpublished)} анализов для публикации")

                        for item in unpublished:
                            # Минимальная проверка перед публикацией
                            validation = self.validator.validate_analysis(item.analysis, item.news.title)

                            # Публикуем ВСЕ анализы, которые не были явно отклонены
                            if validation["is_valid"]:
                                try:
                                    analysis_to_publish = item.analysis
                                    if self.news_guard is not None:
                                        guard_result = self.news_guard.guard_output(analysis=item.analysis)
                                        analysis_to_publish = scrub_after_output_guard(
                                            guard_result.moderated_text
                                        )
                                    success = await self.publisher.publish_analysis(
                                        item.news_id,
                                        item.news.title,
                                        item.news.url,
                                        analysis_to_publish
                                    )
                                    if success:
                                        await repo.mark_as_published(item.news_id)
                                        self.stats["analyses_published"] += 1
                                        logger.info(f"Анализ {item.news_id} успешно опубликован")

                                        # Задержка между публикациями (30 секунд)
                                        await asyncio.sleep(30)
                                    else:
                                        logger.warning(f"Неудачная попытка публикации анализа {item.news_id}")
                                except Exception as e:
                                    logger.error(f"Ошибка публикации анализа {item.news_id}: {str(e)}")
                            else:
                                # Для отклоненных анализов просто помечаем как обработанные
                                logger.warning(f"Анализ {item.news_id} отклонен: {', '.join(validation['reasons'])}")
                                await repo.mark_as_processed_without_analysis(item.news_id)
                                self.stats["analyses_rejected"] += 1

                # Короткая пауза перед следующей проверкой
                await asyncio.sleep(15)

            except Exception as e:
                logger.error(f"Ошибка в цикле публикации: {str(e)}")
                await asyncio.sleep(30)

    async def report_cycle(self):
        """Цикл отчетности (запускается каждые 30 минут)"""
        while True:
            try:
                # Формируем отчет
                report = (
                    f"📊 Отчет за последние 30 минут: "
                    f"Новостей: {self.stats['news_fetched']}, "
                    f"Обработано: {self.stats['news_processed']}, "
                    f"Пропущено: {self.stats['news_skipped']}, "
                    f"Опубликовано: {self.stats['analyses_published']}, "
                    f"Отклонено: {self.stats['analyses_rejected']}, "
                    f"Ошибок: {self.stats['errors']}"
                )

                logger.info(report)
                await self.publisher.send_admin_notification(report)

                # Сброс статистики
                for key in self.stats:
                    self.stats[key] = 0

                # Ждем 30 минут до следующего отчета
                await asyncio.sleep(1800)

            except Exception as e:
                logger.error(f"Ошибка в цикле отчетности: {str(e)}")
                await asyncio.sleep(1800)  # Ждем перед повторной попыткой

    async def start_separated_processing(self):
        """Запуск раздельных циклов обработки"""
        logger.info("Запуск раздельных циклов обработки")
        await self.publisher.send_admin_notification("🔄 Запуск раздельных циклов обработки")

        # Запускаем все циклы параллельно
        await asyncio.gather(
            self.fetch_news_cycle(),
            self.process_news_cycle(),
            self.publish_cycle(),
            self.report_cycle()
        )

    @handle_errors
    async def close(self):
        """Закрытие ресурсов"""
        await self.publisher.send_admin_notification("🛑 Завершение работы системы ИИ-Ленин")

        if self.analyzer:
            await self.analyzer.close_session()
        await self.server.stop_server()

        await self.publisher.send_admin_notification("✅ Система успешно остановлена")