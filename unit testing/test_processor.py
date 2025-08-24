import logging
import asyncio
from src.core.processor import NewsProcessor
from src.core.settings.log import setup_logging


async def test_processor():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Тестирование процессора новостей")

    processor = NewsProcessor()
    logger.info("Запуск одного цикла обработки")
    await processor.run_full_cycle()
    logger.info("Цикл обработки завершен")


if __name__ == "__main__":
    asyncio.run(test_processor())