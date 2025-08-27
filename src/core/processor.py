import asyncio
import time
import logging
from src.core.database.repositories.news_repository import NewsRepository
from src.modules.news_system.fetcher import NewsFetcher
from src.core.lenin_analyzer import LeninAnalyzer
from src.core.publisher import TelegramPublisher
from src.core.settings.config import Settings
from src.core.utils.decorators import handle_errors
from src.core.database.db_core import session_scope
from src.core.llama_server import LeninServer
from src.core.rag_system import get_rag_system
from src.modules.news_system.classifier import NewsClassifier
from src.core.analysis_validator import AnalysisValidator

logger = logging.getLogger(__name__)


class NewsProcessor:
    def __init__(self):
        self.config = Settings()
        logger.info("Инициализация EnhancedNewsProcessor")

        self.fetcher = NewsFetcher()
        self.analyzer = None
        self.server = LeninServer()
        self.classifier = NewsClassifier()
        self.validator = AnalysisValidator()
        self.analyzer_ready = asyncio.Event()

        # Инициализация RAG системы
        self.rag_system = None
        asyncio.create_task(self.initialize_components())

        self.publisher = TelegramPublisher()
        self.stats = {
            "news_fetched": 0,
            "news_processed": 0,
            "news_skipped": 0,
            "analyses_published": 0,
            "analyses_rejected": 0,
            "errors": 0
        }

    async def initialize_rag_system(self):
        """Инициализация RAG системы"""
        try:
            loop = asyncio.get_event_loop()
            self.rag_system = await loop.run_in_executor(None, get_rag_system)
            logger.info("RAG система инициализирована")
            await self.publisher.send_admin_notification("✅ RAG система инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации RAG: {str(e)}")
            await self.publisher.send_admin_notification(f"❌ Ошибка RAG системы: {str(e)[:200]}")

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

            # Инициализация RAG системы
            await self.publisher.send_admin_notification("📚 Инициализация RAG системы...")
            rag_task = asyncio.create_task(self.initialize_rag_system())

            # Инициализация анализатора
            await self.publisher.send_admin_notification("🧠 Инициализация анализатора...")
            self.analyzer = LeninAnalyzer()
            await self.analyzer.initialize_session()
            await self.publisher.send_admin_notification("✅ Анализатор инициализирован")

            # Ожидание завершения инициализации RAG
            await rag_task

            self.analyzer_ready.set()
            logger.info("Все компоненты инициализированы")

            # Отправляем уведомление о успешном запуске
            await self.publisher.send_admin_notification("🚀 Система ИИ-Ленин успешно запущена и готова к работе!")

        except Exception as e:
            logger.exception(f"Ошибка инициализации: {str(e)}")
            await self.publisher.send_admin_notification(
                f"❌ Критическая ошибка инициализации: {str(e)[:300]}"
            )
            raise

    @handle_errors
    async def process_pending_news(self):
        """Обработка ожидающих новостей с улучшенной фильтрацией"""
        if not self.analyzer_ready.is_set():
            logger.info("Ожидание инициализации анализатора...")
            await self.analyzer_ready.wait()

        try:
            async with session_scope() as session:
                repo = NewsRepository(session)
                unprocessed = await repo.get_unprocessed_news(limit=10)

                logger.info(f"Найдено {len(unprocessed)} необработанных новостей")

                if unprocessed:
                    await self.publisher.send_admin_notification(
                        f"📰 Найдено {len(unprocessed)} новостей для обработки"
                    )

                for news in unprocessed:
                    # Логируем информацию о новости
                    logger.info(f"Обработка новости: {news.title[:50]}...")

                    # Проверяем, нужно ли анализировать новость
                    should_analyze, reason = self.classifier.should_analyze(news.title, news.content)
                    logger.info(f"Решение по новости {news.id}: {reason}")

                    if not should_analyze:
                        logger.info(f"Пропуск новости {news.id}: {reason}")
                        await repo.mark_as_processed_without_analysis(news.id)
                        self.stats["news_skipped"] += 1
                        continue

                    try:
                        # Генерируем анализ
                        logger.info(f"Генерация анализа для новости {news.id}")
                        await self.publisher.send_admin_notification(
                            f"🔍 Анализ новости: {news.title[:50]}..."
                        )

                        analysis = await self.analyzer.generate_analysis(
                            news.title,
                            news.content
                        )

                        logger.info(f"Сгенерирован анализ длиной {len(analysis)} символов")

                        # Валидируем анализ
                        validation = self.validator.validate_analysis(analysis, news.title)
                        logger.info(f"Результат валидации: {validation}")

                        if validation["is_valid"]:
                            await repo.save_analysis(news.id, analysis)
                            self.stats["news_processed"] += 1
                            logger.info(f"Успешный анализ новости {news.id}. Оценка: {validation['score']:.2f}")
                            await self.publisher.send_admin_notification(
                                f"✅ Анализ завершен. Оценка: {validation['score']:.2f}"
                            )
                        else:
                            logger.warning(f"Анализ новости {news.id} отклонен: {', '.join(validation['reasons'])}")
                            await repo.mark_as_processed_without_analysis(news.id)
                            self.stats["analyses_rejected"] += 1
                            await self.publisher.send_admin_notification(
                                f"❌ Анализ отклонен: {', '.join(validation['reasons'])[:100]}"
                            )

                    except Exception as e:
                        logger.error(f"Ошибка обработки новости {news.id}: {str(e)}")
                        self.stats["errors"] += 1
                        await self.publisher.send_admin_notification(
                            f"⚠️ Ошибка обработки новости: {str(e)[:200]}"
                        )
        except Exception as e:
            logger.error(f"Ошибка в цикле обработки: {str(e)}")
            self.stats["errors"] += 1
            await self.publisher.send_admin_notification(
                f"❌ Ошибка цикла обработки: {str(e)[:200]}"
            )

    @handle_errors
    async def fetch_new_news(self):
        """Сбор новых новостей"""
        try:
            await self.publisher.send_admin_notification("📡 Начало сбора новостей...")
            news_items = self.fetcher.fetch_all()

            if news_items:
                await self.publisher.send_admin_notification(
                    f"📊 Собрано {len(news_items)} новостей из TASS"
                )

            async with session_scope() as session:
                repo = NewsRepository(session)
                await repo.save_news(news_items)
            self.stats["news_fetched"] += len(news_items)
        except Exception as e:
            logger.error(f"Ошибка при сборе новостей: {str(e)}")
            self.stats["errors"] += 1
            await self.publisher.send_admin_notification(f"❌ Ошибка сбора новостей: {str(e)[:200]}")

    @handle_errors
    async def publish_pending_analyses(self):
        """Публикация готовых анализов с дополнительной проверкой"""
        try:
            async with session_scope() as session:
                repo = NewsRepository(session)
                unpublished = await repo.get_unpublished_analysis(limit=10)

                if unpublished:
                    await self.publisher.send_admin_notification(
                        f"📤 Найдено {len(unpublished)} анализов для публикации"
                    )

                for item in unpublished:
                    # Дополнительная проверка перед публикацией
                    validation = self.validator.validate_analysis(item.analysis, item.news.title)

                    if validation["is_valid"] and validation["score"] > 0.5:
                        for attempt in range(3):  # 3 попытки публикации
                            try:
                                await self.publisher.send_admin_notification(
                                    f"📨 Публикация анализа (попытка {attempt + 1})..."
                                )

                                success = await self.publisher.publish_analysis(
                                    item.news_id,
                                    item.news.title,
                                    item.news.url,
                                    item.analysis
                                )
                                if success:
                                    await repo.mark_as_published(item.news_id)
                                    self.stats["analyses_published"] += 1
                                    await self.publisher.send_admin_notification(
                                        "✅ Анализ успешно опубликован в канале"
                                    )
                                    break
                                else:
                                    await asyncio.sleep(2)
                            except Exception as e:
                                logger.error(f"Ошибка публикации (попытка {attempt + 1}): {str(e)}")
                                await self.publisher.send_admin_notification(
                                    f"⚠️ Ошибка публикации: {str(e)[:200]}"
                                )
                                await asyncio.sleep(2)
                    else:
                        logger.warning(f"Анализ {item.news_id} не прошел финальную проверку")
                        await repo.mark_as_processed_without_analysis(item.news_id)
                        self.stats["analyses_rejected"] += 1
                        await self.publisher.send_admin_notification(
                            f"❌ Анализ не прошел финальную проверку. Оценка: {validation['score']:.2f}"
                        )
        except Exception as e:
            logger.error(f"Ошибка в цикле публикации: {str(e)}")
            self.stats["errors"] += 1
            await self.publisher.send_admin_notification(
                f"❌ Ошибка цикла публикации: {str(e)[:200]}"
            )

    async def run_enhanced_cycle(self):
        """Улучшенный цикл обработки с фильтрацией и валидацией"""
        logger.info("Запуск улучшенного цикла обработки")
        await self.publisher.send_admin_notification("🔄 Запуск цикла обработки")

        await self.fetch_new_news()
        await self.process_pending_news()
        await self.publish_pending_analyses()

        # Формируем отчет
        report = (
            f"📊 Цикл завершен: "
            f"Новостей: {self.stats['news_fetched']}, "
            f"Обработано: {self.stats['news_processed']}, "
            f"Пропущено: {self.stats['news_skipped']}, "
            f"Опубликовано: {self.stats['analyses_published']}, "
            f"Отклонено: {self.stats['analyses_rejected']}, "
            f"Ошибок: {self.stats['errors']}"
        )

        logger.info(report)

        # Отправляем отчет администратору
        await self.publisher.send_admin_notification(report)

        # Сброс статистики
        for key in self.stats:
            self.stats[key] = 0

    async def start_optimized_processing(self):
        """Запуск оптимизированной обработки"""
        logger.info("Запуск оптимизированной обработки")

        # Первый цикл
        await self.run_enhanced_cycle()

        # Адаптивный основной цикл
        while True:
            start_time = time.time()
            await self.run_enhanced_cycle()
            elapsed = time.time() - start_time

            # Адаптивная пауза based на времени выполнения
            target_cycle_time = 300  # 5 минут
            sleep_time = max(60, target_cycle_time - elapsed)  # Не менее 60 секунд

            logger.info(f"Ожидание {sleep_time} сек. до следующего цикла")
            await self.publisher.send_admin_notification(
                f"⏰ Ожидание {sleep_time} сек. до следующего цикла"
            )

            await asyncio.sleep(sleep_time)

    @handle_errors
    async def close(self):
        """Закрытие ресурсов"""
        await self.publisher.send_admin_notification("🛑 Завершение работы системы ИИ-Ленин")

        if self.analyzer:
            await self.analyzer.close_session()
        await self.server.stop_server()

        await self.publisher.send_admin_notification("✅ Система успешно остановлена")