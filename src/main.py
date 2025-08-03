import sys
import os
import asyncio
import logging
import platform
import torch
from src.core.settings.log import setup_logging
from src.core.database.db_migrations import apply_migrations
from src.core.processor import NewsProcessor
from src.core.database.db_core import session_scope
from sqlalchemy import text

setup_logging()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

async def async_main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Запуск системы ИИ-Ленин")

    try:
        # Диагностика GPU
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"Доступно VRAM: {total_vram:.2f} GB")
        else:
            logger.warning("CUDA недоступна! Работа на CPU будет очень медленной")

        # Проверка переменных окружения
        required_envs = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHANNEL_ID", "TELEGRAM_ADMIN_ID"]
        missing = [env for env in required_envs if not os.getenv(env)]
        if missing:
            logger.error(f"Отсутствуют переменные окружения: {', '.join(missing)}")
            sys.exit(1)
        else:
            logger.info("Все необходимые переменные окружения найдены")

        # Применение миграций
        logger.info("Применение миграций БД")
        if not await apply_migrations():
            logger.error("Не удалось применить миграции БД. Система остановлена.")
            sys.exit(1)
        else:
            logger.info("Миграции БД успешно применены")

        # Проверка соединения с БД
        try:
            async with session_scope() as session:
                logger.info("Проверка соединения с SQLite базой данных")
                result = await session.execute(text("SELECT 1"))
                if result.scalar() == 1:
                    logger.info("Соединение с SQLite успешно установлено")
                else:
                    logger.error("Ошибка проверки SQLite соединения")
        except Exception as e:
            logger.error(f"Ошибка подключения к SQLite: {str(e)}")
            sys.exit(1)

        # Инициализация процессора новостей
        logger.info("Инициализация процессора новостей")
        processor = NewsProcessor()

        # Даем время на запуск потока инициализации
        await asyncio.sleep(1)
        logger.info("Процессор успешно инициализирован")

        # Запуск основного цикла
        logger.info("Запуск основного цикла обработки")
        await processor.start_periodic_processing()

    except Exception as e:
        logger.exception(f"Критическая ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Фикс для Windows
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("Приложение остановлено пользователем")
    except Exception as e:
        print(f"Необработанная ошибка: {str(e)}")