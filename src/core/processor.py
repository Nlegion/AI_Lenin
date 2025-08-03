# === src/core/processor.py ===
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
        self.analyzer = None
        self.analyzer_ready = asyncio.Event()
        asyncio.create_task(self.initialize_analyzer_async())
        logger.info("Инициализация TelegramPublisher")
        self.publisher = TelegramPublisher()
        self.stats = {"news_fetched": 0, "news_processed": 0, "analyses_published": 0, "errors": 0}

    @handle_errors
    async def initialize_analyzer_async(self):
        try:
            self.analyzer = LeninAnalyzer()
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.exception(f"Ошибка инициализации: {str(e)}")
            await self.publisher.send_admin_notification(f"🚨 Ошибка загрузки модели: {str(e)[:300]}")
        finally:
            self.analyzer_ready.set()

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
                unprocessed = await repo.get_unprocessed_news(limit=1)  # По одной новости

                for news in unprocessed:
                    # Проверка доступной VRAM
                    if torch.cuda.is_available() and torch.cuda.memory_allocated() > 6.5e9:
                        logger.warning("Превышение лимита VRAM, пропуск новости")
                        continue
                    try:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        loop = asyncio.get_running_loop()
                        analysis = await loop.run_in_executor(
                            None,
                            self.analyzer.generate_analysis,
                            news.title,  # Передаем заголовок
                            news.content  # Передаем содержание
                        )
                        await repo.save_analysis(news.id, analysis)
                        self.stats["news_processed"] += 1
                        # После генерации проверяем, не превысили ли лимит VRAM
                        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 7e9:
                            logger.warning("Превышение лимита VRAM после генерации, выгрузка модели")
                            await loop.run_in_executor(None, self.analyzer.unload_model)
                            await asyncio.sleep(3)  # Пауза для освобождения памяти
                            await loop.run_in_executor(None, self.analyzer.reload_model)
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
            sleep_time = max(300, self.config.UPDATE_INTERVAL - elapsed)  # Минимум 5 минут
            logger.info(f"Ожидание {sleep_time} сек. до следующего цикла")
            await asyncio.sleep(sleep_time)