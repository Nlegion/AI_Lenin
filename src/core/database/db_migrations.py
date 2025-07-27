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
        alembic_cfg = Config(os.path.join(project_root, 'alembic.ini'))
        logger.info(f"Конфиг Alembic: {alembic_cfg.config_file_name}")
        command.upgrade(alembic_cfg, 'head')
        logger.info("Миграции успешно применены")
        return True
    except Exception as e:
        logger.error(f"Ошибка миграций: {str(e)}")
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
        logger.error(f"Ошибка применения миграций: {str(e)}")
        return False