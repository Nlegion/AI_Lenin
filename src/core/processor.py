import asyncio
import time
import logging
from src.core.database.repositories.news_repository import NewsRepository
from src.modules.news_system.fetcher import NewsFetcher
from src.core.lenin_analyzer import EnhancedLeninAnalyzer
from src.core.publisher import TelegramPublisher
from src.core.settings.config import Settings
from src.core.utils.decorators import handle_errors
from src.core.database.db_core import session_scope
from src.core.llama_server import LeninServer
from src.core.rag_system import get_rag_system

logger = logging.getLogger(__name__)


class OptimizedNewsProcessor:
    def __init__(self):
        self.config = Settings()
        logger.info("Инициализация OptimizedNewsProcessor")

        self.fetcher = NewsFetcher()
        self.analyzer = None
        self.server = LeninServer()
        self.analyzer_ready = asyncio.Event()

        # Параллельная инициализация
        asyncio.create_task(self.initialize_components())

        self.publisher = TelegramPublisher()
        self.stats = {
            "news_fetched": 0,
            "news_processed": 0,
            "analyses_published": 0,
            "errors": 0,
            "avg_processing_time": 0
        }
        self.processing_times = []

    async def initialize_components(self):
        """Параллельная инициализация компонентов"""
        try:
            # Инициализация RAG системы в отдельной задаче
            rag_task = asyncio.create_task(self.initialize_rag())

            # Инициализация анализатора
            self.analyzer = EnhancedLeninAnalyzer()
            await self.analyzer.initialize_session()

            # Ожидание завершения инициализации RAG
            await rag_task

            self.analyzer_ready.set()
            logger.info("Все компоненты инициализированы")

        except Exception as e:
            logger.exception(f"Ошибка инициализации: {str(e)}")
            await self.publisher.send_admin_notification(
                f"🚨 Критическая ошибка инициализации: {str(e)[:300]}"
            )

    async def initialize_rag(self):
        """Инициализация RAG системы"""
        try:
            rag_system = get_rag_system()
            # Проверяем, нужно ли перестроить индекс
            if rag_system.collection.count() == 0:
                logger.info("Построение индекса онтологии...")
                await rag_system.build_ontology_index()
            logger.info("RAG система готова")
        except Exception as e:
            logger.error(f"Ошибка инициализации RAG: {str(e)}")

    @handle_errors
    async def process_pending_news(self):
        if not self.analyzer_ready.is_set():
            logger.info("Ожидание инициализации анализатора...")
            await self.analyzer_ready.wait()

        try:
            async with session_scope() as session:
                repo = NewsRepository(session)
                unprocessed = await repo.get_unprocessed_news(limit=5)  # Увеличили лимит

                for news in unprocessed:
                    start_time = time.time()
                    try:
                        analysis = await self.analyzer.generate_analysis(
                            news.title,
                            news.content
                        )

                        if self._is_quality_analysis(analysis):
                            await repo.save_analysis(news.id, analysis)
                            self.stats["news_processed"] += 1

                            # Замер времени обработки
                            processing_time = time.time() - start_time
                            self.processing_times.append(processing_time)
                            self.stats["avg_processing_time"] = sum(
                                self.processing_times[-10:]
                            ) / min(10, len(self.processing_times))

                        else:
                            logger.warning(f"Низкое качество анализа, пропускаем новость {news.id}")
                            await repo.mark_as_processed_without_analysis(news.id)

                    except Exception as e:
                        logger.error(f"Ошибка обработки новости {news.id}: {str(e)}")
                        self.stats["errors"] += 1
        except Exception as e:
            logger.error(f"Ошибка в цикле обработки: {str(e)}")
            self.stats["errors"] += 1

    def _is_quality_analysis(self, analysis: str) -> bool:
        """Улучшенная проверка качества анализа"""
        if not analysis or len(analysis) < 40:
            return False

        # Проверяем на наличие шаблонных фраз
        template_phrases = [
            "теперь", "рассмотрим", "анализируя",
            "можно сделать вывод", "данная ситуация",
            "в контексте новости", "как отмечал"
        ]

        text_lower = analysis.lower()
        if any(phrase in text_lower for phrase in template_phrases):
            return False

        # Проверяем наличие марксистской терминологии
        marxist_terms = [
            "класс", "капитал", "пролетариат", "буржуазия",
            "эксплуатация", "противоречие", "диалектика"
        ]

        if not any(term in text_lower for term in marxist_terms):
            return False

        # Проверяем, что это законченные предложения
        if analysis.count('.') < 1:
            return False

        return True

    @handle_errors
    async def run_optimized_cycle(self):
        """Оптимизированный цикл обработки"""
        logger.info("Запуск оптимизированного цикла обработки")

        # Параллельное выполнение задач
        fetch_task = asyncio.create_task(self.fetch_new_news())
        process_task = asyncio.create_task(self.process_pending_news())

        await asyncio.gather(fetch_task, process_task)
        await self.publish_pending_analyses()

        logger.info(
            f"Цикл завершен: Новостей: {self.stats['news_fetched']}, "
            f"Обработано: {self.stats['news_processed']}, "
            f"Среднее время: {self.stats['avg_processing_time']:.2f}с, "
            f"Ошибок: {self.stats['errors']}"
        )

    async def start_optimized_processing(self):
        """Запуск оптимизированной обработки"""
        logger.info("Запуск оптимизированной обработки")
        await self.publisher.send_admin_notification("🚀 Система ИИ-Ленин запущена (оптимизированная версия)")

        # Первый цикл
        await self.run_optimized_cycle()

        # Адаптивный основной цикл
        while True:
            start_time = time.time()
            await self.run_optimized_cycle()
            elapsed = time.time() - start_time

            # Адаптивная пауза based на времени выполнения
            target_cycle_time = 180
            sleep_time = max(30, target_cycle_time - elapsed)  # Не менее 30 секунд

            logger.info(f"Ожидание {sleep_time} сек. до следующего цикла")
            await asyncio.sleep(sleep_time)