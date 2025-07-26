import structlog
from functools import wraps

logger = structlog.get_logger()

def handle_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Ошибка в {func.__name__}: {str(e)}")
            raise
    return wrapper

def handle_db_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Ошибка БД в {func.__name__}: {str(e)}")
            raise
    return wrapper