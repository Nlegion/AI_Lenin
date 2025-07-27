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
import gc
import torch


logger = logging.getLogger(__name__)


class NewsProcessor:
    def __init__(self):
        self.config = Settings()
        logger.info("Инициализация NewsFetcher")
        self.fetcher = NewsFetcher()

        logger.info("Инициализация LeninAnalyzer")
        try:
            self.analyzer = LeninAnalyzer()
            logger.info("LeninAnalyzer успешно инициализирован")
        except Exception as e:
            logger.exception(f"Ошибка инициализации LeninAnalyzer: {str(e)}")
            raise

        logger.info("Инициализация TelegramPublisher")
        self.publisher = TelegramPublisher()
        logger.info("TelegramPublisher инициализирован")

        logger.info("Инициализация NewsProcessor завершена")
        self.stats = {
            "news_fetched": 0,
            "news_processed": 0,
            "analyses_published": 0,
            "errors": 0
        }

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
        try:
            async with session_scope() as session:
                repo = NewsRepository(session)
                unprocessed = await repo.get_unprocessed_news(limit=1)  # Только 1 новость за раз

                for news in unprocessed:
                    try:
                        # Очистка памяти перед обработкой
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        loop = asyncio.get_running_loop()
                        analysis = await loop.run_in_executor(
                            None,
                            self.analyzer.generate_analysis,
                            news
                        )
                        await repo.save_analysis(news.id, analysis)
                        self.stats["news_processed"] += 1

                        # Пауза для охлаждения GPU
                        await asyncio.sleep(10)
                    except Exception as e:
                        logger.error(f"Ошибка обработки новости {news.id}: {str(e)}")
                        self.stats["errors"] += 1
        except Exception as e:
            logger.error(f"Ошибка в цикле обработки: {str(e)}")
            self.stats["errors"] += 1
            await self.publisher.send_admin_notification(f"🚨 Ошибка обработки: {str(e)}")

    @handle_errors
    async def publish_pending_analyses(self):
        try:
            async with session_scope() as session:
                repo = NewsRepository(session)
                unpublished = await repo.get_unpublished_analysis(limit=5)

                for item in unpublished:
                    try:
                        success = await self.publisher.publish_analysis(
                            item.news_id,
                            item.news.title,
                            item.news.url,
                            item.analysis
                        )
                        if success:
                            await repo.mark_as_published(item.news_id)
                            self.stats["analyses_published"] += 1
                        else:
                            logger.error(f"Не удалось опубликовать анализ: {item.news_id}")
                            self.stats["errors"] += 1
                    except Exception as e:
                        logger.error(f"Ошибка публикации: {str(e)}")
                        self.stats["errors"] += 1
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
        try:
            await self.publisher.send_admin_notification("🚀 Система ИИ-Ленин запущена")
            logger.info("Уведомление администратору отправлено")

            # Первоначальная обработка
            logger.info("Запуск первого цикла обработки")
            await self.run_full_cycle()
            logger.info("Первый цикл обработки завершен")

            # Основной цикл
            while True:
                logger.info("Начало нового цикла обработки")
                start_time = time.time()
                await self.run_full_cycle()
                elapsed = time.time() - start_time
                sleep_time = max(1, self.config.UPDATE_INTERVAL - elapsed)
                logger.info(f"Цикл завершен за {elapsed:.2f} сек. Ожидание {sleep_time} сек.")
                await asyncio.sleep(sleep_time)

        except Exception as e:
            logger.exception(f"Ошибка в основном цикле: {str(e)}")
            await self.publisher.send_admin_notification(f"🛑 Критическая ошибка: {str(e)[:500]}")
            raise