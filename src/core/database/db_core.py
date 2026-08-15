from contextlib import asynccontextmanager
import os
import structlog
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

logger = structlog.get_logger()

Base = declarative_base()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATABASE_URL = f"sqlite+aiosqlite:///{os.path.join(project_root, 'ai_lenin.db')}"

# Увеличиваем таймауты и настраиваем пул соединений
engine = create_async_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30,  # Увеличиваем таймаут до 30 секунд
    },
    echo=False,  # Отключаем подробное логирование SQL
    logging_name="sqlalchemy.engine",
    pool_size=10,  # Увеличиваем размер пула
    max_overflow=20,  # Разрешаем больше переполнений
    pool_timeout=30,  # Таймаут ожидания соединения из пула
)

# Настраиваем логгинг SQLAlchemy
logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.pool").setLevel(logging.ERROR)

async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def session_scope():
    session = async_session()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error("Database session error", error=str(e), exc_info=True)
        raise
    finally:
        if session:
            try:
                await session.close()
            except Exception as e:
                logger.error("Error closing session", error=str(e))
