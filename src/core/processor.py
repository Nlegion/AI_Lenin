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

logger = logging.getLogger(__name__)


class NewsProcessor:
    def __init__(self):
        self.config = Settings()
        logger.info("Инициализация NewsFetcher")
        self.fetcher = NewsFetcher()
        self.analyzer = None
        self.server = LeninServer()  # Добавляем сервер
        self.analyzer_ready = asyncio.Event()
        asyncio.create_task(self.initialize_analyzer_async())
        logger.info("Инициализация TelegramPublisher")
        self.publisher = TelegramPublisher()
        self.stats = {"news_fetched": 0, "news_processed": 0, "analyses_published": 0, "errors": 0}

    @handle_errors
    async def initialize_analyzer_async(self):
        try:
            # Запускаем сервер
            if not await self.server.start_server():
                raise Exception("Не удалось запустить сервер llama.cpp")

            # Инициализируем анализатор
            self.analyzer = LeninAnalyzer()
            await self.analyzer.initialize_session()

        except Exception as e:
            logger.exception(f"Ошибка инициализации: {str(e)}")
            await self.publisher.send_admin_notification(f"🚨 Ошибка загрузки модели: {str(e)[:300]}")
        finally:
            self.analyzer_ready.set()

    @handle_errors
    async def close(self):
        """Закрытие ресурсов"""
        if self.analyzer:
            await self.analyzer.close_session()
        await self.server.stop_server()

    @handle_errors
    async def fetch_new_news(self):
        try:
            news_items = self.fetcher.fetch_all()
            async with session_scope() as session:
                repo = NewsRepository(session)
                await repo.save_news(news_items)
            self.stats["news_fetched"] += len(news_items)
        except Exception as e:
            logger.error(f"Ошибка при сборе новостей: {str(e)}")
            self.stats["errors"] += 1
            await self.publisher.send_admin_notification(f"🚨 Ошибка сбора новостей: {str(e)}")

    @handle_errors
    async def process_pending_news(self):
        if not self.analyzer_ready.is_set():
            logger.info("Ожидание инициализации анализатора...")
            await self.analyzer_ready.wait()

        try:
            async with session_scope() as session:
                repo = NewsRepository(session)
                # Обрабатываем больше новостей за цикл
                unprocessed = await repo.get_unprocessed_news(limit=3)

                for news in unprocessed:
                    try:
                        analysis = await self.analyzer.generate_analysis(
                            news.title,
                            news.content
                        )

                        # Проверяем качество анализа перед сохранением
                        if self._is_quality_analysis(analysis):
                            await repo.save_analysis(news.id, analysis)
                            self.stats["news_processed"] += 1
                        else:
                            logger.warning(f"Низкое качество анализа, пропускаем новость {news.id}")
                            # Помечаем как обработанную, но без анализа
                            await repo.mark_as_processed_without_analysis(news.id)

                    except Exception as e:
                        logger.error(f"Ошибка обработки новости {news.id}: {str(e)}")
                        self.stats["errors"] += 1
        except Exception as e:
            logger.error(f"Ошибка в цикле обработки: {str(e)}")
            self.stats["errors"] += 1

    def _is_quality_analysis(self, analysis: str) -> bool:
        if not analysis or len(analysis) < 30:
            return False

        # Проверяем на наличие шаблонных фраз
        template_phrases = [
            "теперь", "рассмотрим", "анализируя",
            "можно сделать вывод", "данная ситуация",
            "в контексте новости"
        ]

        if any(phrase in analysis.lower() for phrase in template_phrases):
            return False

        # Проверяем, что это законченные предложения
        if analysis.count('.') < 1:  # Хотя бы одна точка
            return False

        return True

    @handle_errors
    async def publish_pending_analyses(self):
        try:
            async with session_scope() as session:
                repo = NewsRepository(session)
                unpublished = await repo.get_unpublished_analysis(limit=10)  # Увеличили лимит

                for item in unpublished:
                    for attempt in range(3):  # 3 попытки
                        try:
                            success = await self.publisher.publish_analysis(
                                item.news_id,
                                item.news.title,
                                item.news.url,
                                item.analysis
                            )
                            if success:
                                await repo.mark_as_published(item.news_id)
                                break  # Выход при успехе
                            else:
                                await asyncio.sleep(2)  # Пауза между попытками
                        except Exception as e:
                            logger.error(f"Ошибка публикации (попытка {attempt + 1}): {str(e)}")
        except Exception as e:
            logger.error(f"Ошибка в цикле публикации: {str(e)}")
            self.stats["errors"] += 1
            await self.publisher.send_admin_notification(f"🚨 Ошибка публикации: {str(e)}")

    @handle_errors
    async def run_full_cycle(self):
        logger.info("Запуск полного цикла обработки")
        await self.fetch_new_news()
        await self.process_pending_news()
        await self.publish_pending_analyses()

        logger.info(
            f"Цикл завершен: Новостей: {self.stats['news_fetched']}, "
            f"Обработано: {self.stats['news_processed']}, "
            f"Опубликовано: {self.stats['analyses_published']}, "
            f"Ошибок: {self.stats['errors']}"
        )

        # Отчет администратору только при ошибках
        if self.stats['errors'] > 0:
            await self.publisher.send_admin_notification(
                f"⚠️ Цикл завершен с {self.stats['errors']} ошибками. Проверьте логи."
            )

        # Сброс статистики
        for key in self.stats:
            self.stats[key] = 0

    @handle_errors
    async def start_periodic_processing(self):
        logger.info("Запуск периодической обработки")
        await self.publisher.send_admin_notification("🚀 Система ИИ-Ленин запущена")

        # Первый цикл
        await self.run_full_cycle()

        # Основной цикл
        while True:
            start_time = time.time()
            await self.run_full_cycle()
            elapsed = time.time() - start_time

            # ФИКС: Максимальное время ожидания - 5 минут (300 секунд)
            sleep_time = min(300, max(60, 300 - elapsed))  # От 1 до 5 минут

            logger.info(f"Ожидание {sleep_time} сек. до следующего цикла")
            await asyncio.sleep(sleep_time)