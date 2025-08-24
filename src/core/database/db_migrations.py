import os
import logging
import structlog
from alembic import command
from alembic.config import Config
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)


def run_migrations():
    logger.info("Запуск синхронных миграций Alembic")
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

        # Полный путь к папке миграций
        migrations_path = os.path.join(project_root, "src", "core", "database", "migrations")
        logger.info(f"Путь к миграциям: {migrations_path}")

        # Создаем конфиг Alembic
        alembic_cfg = Config()
        alembic_cfg.set_main_option("script_location", migrations_path)
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{os.path.join(project_root, 'ai_lenin.db')}")

        # Настройка логирования Alembic
        alembic_cfg.attributes['configure_logger'] = False
        logging.getLogger('alembic').setLevel(logging.WARNING)

        command.upgrade(alembic_cfg, 'head')
        logger.info("Миграции успешно применены")
        return True
    except Exception as e:
        logger.exception(f"Ошибка миграций: {str(e)}")
        return False


async def apply_migrations():
    logger.info("Применение миграций БД")
    try:
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(ThreadPoolExecutor(), run_migrations)
        if success:
            logger.info("Миграции БД применены успешно")
        else:
            logger.error("Не удалось применить миграции")
        return success
    except Exception as e:
        logger.exception(f"Ошибка применения миграций: {str(e)}")
        return False