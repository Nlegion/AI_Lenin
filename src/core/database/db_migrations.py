import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from alembic import command
from alembic.config import Config
import structlog

logger = structlog.get_logger()


def run_migrations():
    # Получаем абсолютный путь к корневой директории проекта
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    alembic_cfg = Config(os.path.join(project_root, 'alembic.ini'))
    command.upgrade(alembic_cfg, 'head')

async def apply_migrations():
    logger.info("Applying database migrations")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(ThreadPoolExecutor(), run_migrations)
        logger.info("Migrations applied successfully")
    except Exception as e:
        logger.error("Migration failed", error=str(e))
        raise
