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
from src.core.version import version_manager
from src.core.settings.config import Settings
from src.core.settings.generation_config import (
    default_generation_config_path,
    llm_spawn_local_from_env,
    load_generation_config,
)
from src.core.retrieval.rag_preflight import RagPreflightError, run_rag_preflight
from sqlalchemy import text
from pathlib import Path

# Настройка логирования SQLAlchemy на уровень ERROR
logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.pool").setLevel(logging.ERROR)

setup_logging()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


async def async_main():
    logger = logging.getLogger(__name__)
    version_info = version_manager.get_full_version()
    logger.info(f"Запуск системы ИИ-Ленин {version_info} с раздельными циклами")

    processor = None
    try:
        # Диагностика GPU
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Доступно VRAM: {total_vram:.2f} GB")
        else:
            logger.warning("CUDA недоступна! Работа на CPU будет медленнее")

        # Проверка переменных окружения
        required_envs = [
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHANNEL_ID",
            "TELEGRAM_ADMIN_ID",
        ]
        missing = [env for env in required_envs if not os.getenv(env)]
        if missing:
            logger.error(f"Отсутствуют переменные окружения: {', '.join(missing)}")
            sys.exit(1)
        else:
            logger.info("Все необходимые переменные окружения найдены")

        # Fail-fast generation/provider config before RAG preflight and processor init
        try:
            generation_config = load_generation_config(
                path=default_generation_config_path(base_dir=Path(Settings.BASE_DIR))
            )
            logger.info(
                "Generation provider=%s spawn_local=%s model=%s",
                generation_config.provider,
                generation_config.spawn_local,
                generation_config.active_backend().model_name,
            )
        except ValueError as error:
            logger.error("Invalid generation config: %s", error)
            sys.exit(1)

        if not llm_spawn_local_from_env():
            logger.info("Remote LLM mode: running RAG preflight")
            try:
                run_rag_preflight(base_dir=Path(Settings.BASE_DIR))
            except RagPreflightError as error:
                logger.error("RAG preflight failed: %s", error)
                sys.exit(1)

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

        # Даем время на запуск сервера и инициализацию
        await asyncio.sleep(15)
        logger.info("Процессор успешно инициализирован")

        # Запуск раздельных циклов обработки
        logger.info("Запуск раздельных циклов обработки")
        await processor.start_separated_processing()

    except Exception as e:
        logger.exception(f"Критическая ошибка: {str(e)}")
        if processor:
            await processor.publisher.send_admin_notification(
                f"💥 Критическая ошибка системы: {str(e)[:300]}"
            )
        sys.exit(1)
    finally:
        # Гарантированное закрытие ресурсов
        if processor:
            await processor.close()


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    logger = logging.getLogger(__name__)
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Приложение остановлено пользователем")
    except Exception as e:
        logger.exception("Необработанная ошибка: %s", e)
