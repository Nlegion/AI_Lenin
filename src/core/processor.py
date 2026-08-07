import asyncio
import time
import logging
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
from src.core.safety.news_guard import NewsGuard
from src.core.safety.safety_gate import SafetyGate
from src.core.safety.safety_gate_types import GateContext
from src.core.settings.analysis_defaults import REFUSAL_PHRASES

logger = logging.getLogger(__name__)
db_lock = asyncio.Lock()

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
            "errors": 0
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

        # Семафор для ограничения параллельных обработок
        processing_semaphore = asyncio.Semaphore(2)  # Максимум 2 одновременно

        while True:
            try:
                async with db_lock:
                    async with session_scope() as session:
                        repo = NewsRepository(session)
                        unprocessed = await repo.get_unprocessed_news(limit=5)

                        logger.info(f"Найдено {len(unprocessed)} необработанных новостей")

                        processing_tasks = []
                        for news in unprocessed:
                            # Сбрасываем счетчик попыток для новой новости
                            if news.id in self.failed_attempts:
                                del self.failed_attempts[news.id]

                            # Создаем задачу с ограничением через семафор
                            task = asyncio.create_task(
                                self.process_single_news(news, repo, processing_semaphore)
                            )
                            processing_tasks.append(task)

                        # Ждем завершения всех задач
                        if processing_tasks:
                            await asyncio.gather(*processing_tasks)

                # Короткая пауза перед следующей проверкой новых новостей
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Ошибка в цикле обработки новостей: {str(e)}")
                await self.publisher.send_admin_notification(f"❌ Ошибка обработки новостей: {str(e)[:200]}")
                await asyncio.sleep(30)

    async def process_single_news(self, news, repo, semaphore):
        """Обработка одной новости с ограничением параллелизма"""
        async with semaphore:
            # Логируем информацию о новости
            logger.info(f"Обработка новости: {news.title[:50]}...")

            gate_decision = None
            if self.safety_gate is not None and self.safety_gate.config.flags.enabled:
                ctx = GateContext(
                    title=news.title,
                    content=news.content,
                    source=news.source,
                    item_id=str(news.id),
                    config_version_hash=self.safety_gate.config_version_hash,
                )
                shadow = self.safety_gate.evaluate_with_shadow(
                    ctx,
                    legacy_guard=self.news_guard,
                )
                gate_decision = shadow.enforced
                logger.info(
                    "SafetyGate decision news_id=%s decision=%s risk_tier=%s "
                    "match=%s old=%s new=%s codes=%s",
                    news.id,
                    gate_decision.decision,
                    gate_decision.risk_tier,
                    shadow.decision_match,
                    shadow.old_decision.decision if shadow.old_decision else None,
                    shadow.new_decision.decision if shadow.new_decision else None,
                    ",".join(gate_decision.reason_codes),
                )
            elif self.news_guard is not None:
                legacy = self.news_guard.evaluate_input(
                    news.title, news.content, source=news.source
                )
                from src.core.safety.safety_gate_types import GateDecision, SafetyHint

                hints = []
                if legacy.risk_tier == "yellow":
                    hints = [
                        SafetyHint.YELLOW_CONSTRAINED_ANALYSIS,
                        SafetyHint.AVOID_COMBAT_ESTIMATES,
                    ]
                gate_decision = GateDecision(
                    decision=legacy.decision,
                    risk_tier=legacy.risk_tier,
                    reason=legacy.reason,
                    reason_codes=list(legacy.reason_codes),
                    message=legacy.message,
                    context_hints=hints,
                    needs_yellow_warning=legacy.risk_tier == "yellow"
                    and legacy.decision == "allow",
                )
                logger.info(
                    "NewsGate decision news_id=%s decision=%s risk_tier=%s reason=%s codes=%s",
                    news.id,
                    gate_decision.decision,
                    gate_decision.risk_tier,
                    gate_decision.reason,
                    ",".join(gate_decision.reason_codes),
                )
            if gate_decision is not None:
                if gate_decision.decision in {"deny", "quarantine", "skip"}:
                    await repo.mark_as_processed_without_analysis(news.id)
                    self.stats["news_skipped"] += 1
                    return
                setattr(news, "_risk_tier", gate_decision.risk_tier)
                setattr(
                    news,
                    "_context_hints",
                    [h.value if hasattr(h, "value") else str(h) for h in gate_decision.context_hints],
                )
                setattr(news, "_needs_yellow_warning", gate_decision.needs_yellow_warning)
                if gate_decision.risk_tier == "yellow":
                    from src.core.safety.yellow_audit import append_yellow_audit

                    append_yellow_audit(
                        base_dir=Path(__file__).resolve().parents[2],
                        item_id=str(news.id),
                        title=news.title,
                        content=news.content,
                        risk_tier=gate_decision.risk_tier,
                        reason_codes=list(gate_decision.reason_codes),
                        decision=gate_decision.decision,
                    )

            # Проверяем, нужно ли анализировать новость
            should_analyze, reason = self.classifier.should_analyze(news.title, news.content)
            logger.info(f"Решение по новости {news.id}: {reason}")

            if not should_analyze:
                logger.info(f"Пропуск новости {news.id}: {reason}")
                await repo.mark_as_processed_without_analysis(news.id)
                self.stats["news_skipped"] += 1
                return

            try:
                # Генерируем анализ
                logger.info(f"Генерация анализа для новости {news.id}")
                analysis = await self.analyzer.generate_analysis(
                    news.title,
                    news.content,
                    risk_tier=getattr(news, "_risk_tier", "green"),
                    context_hints=getattr(news, "_context_hints", None),
                    needs_yellow_warning=bool(getattr(news, "_needs_yellow_warning", False)),
                )

                # Проверяем, не отказалась ли модель от анализа
                if any(phrase in analysis.lower() for phrase in REFUSAL_PHRASES):
                    logger.info(f"Модель отказалась анализировать новость {news.id}")
                    await repo.mark_as_processed_without_analysis(news.id)
                    self.stats["news_skipped"] += 1
                    return

                logger.info(f"Сгенерирован анализ длиной {len(analysis)} символов")

                if self.news_guard is not None:
                    guard_result = self.news_guard.guard_output(
                        analysis=analysis,
                        source_text=f"{news.title}\n{news.content}",
                        risk_tier=getattr(news, "_risk_tier", "green"),
                    )
                    logger.info(
                        "NewsGuard output news_id=%s blocked=%s codes=%s",
                        news.id,
                        guard_result.blocked,
                        ",".join(guard_result.reason_codes),
                    )
                    analysis = guard_result.moderated_text

                # Валидируем анализ
                validation = self.validator.validate_analysis(analysis, news.title)
                logger.info(f"Результат валидации: {validation}")

                if validation["is_valid"]:
                    await repo.save_analysis(news.id, analysis)
                    self.stats["news_processed"] += 1
                    logger.info(f"Успешный анализ новости {news.id}. Оценка: {validation['score']:.2f}")
                else:
                    logger.warning(f"Анализ новости {news.id} отклонен: {', '.join(validation['reasons'])}")
                    await repo.mark_as_processed_without_analysis(news.id)
                    self.stats["analyses_rejected"] += 1

            except Exception as e:
                logger.error(f"Ошибка обработки новости {news.id}: {str(e)}")
                self.stats["errors"] += 1

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
                                        analysis_to_publish = guard_result.moderated_text
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