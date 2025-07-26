import asyncio
import os
import sys
import logging
from src.core.settings.log import setup_logging
from src.core.database.db_migrations import apply_migrations
from src.core.processor import NewsProcessor

async def async_main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Запуск системы ИИ-Ленин")

    # Проверка обязательных переменных окружения
    required_envs = [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHANNEL_ID",
        "TELEGRAM_API_ID",
        "TELEGRAM_API_HASH"
    ]
    missing = [env for env in required_envs if not os.getenv(env)]

    if missing:
        logger.error(f"Отсутствуют обязательные переменные окружения: {', '.join(missing)}")
        sys.exit(1)

    # Применяем миграции БД
    await apply_migrations()

    try:
        processor = NewsProcessor()
        await processor.start_periodic_processing()
    except Exception as e:
        logger.exception(f"Критическая ошибка в основном процессе: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(async_main())