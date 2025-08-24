from sqlalchemy import select, update, insert
from sqlalchemy.orm import selectinload
from src.core.database.models.models import News, Analysis
from src.core.utils.decorators import handle_db_errors
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class NewsRepository:
    def __init__(self, session):
        self.session = session
        self.stats = {
            "news_fetched": 0,
            "news_processed": 0,
            "analyses_published": 0,
            "errors": 0
        }

    @handle_db_errors
    async def mark_as_processed_without_analysis(self, news_id: str):
        """Помечает новость как обработанную без сохранения анализа"""
        stmt = update(News).where(
            News.id == news_id
        ).values(
            processed=True,
            processed_at=datetime.utcnow()
        )
        await self.session.execute(stmt)

    @handle_db_errors
    async def save_news(self, news_items: list):
        if not news_items:
            return

        logger.info(f"Сохранение {len(news_items)} новостей в БД")

        # Подготовка данных для пакетной вставки
        data = []
        for item in news_items:
            news_data = {
                "id": item['id'],
                "title": item['title'],
                "content": item['content'],
                "source": item['source'],
                "date": item['date'],
                "url": item['url'],
                "processed": False,
                "processed_at": None,  # Явно указываем NULL
                "created_at": datetime.utcnow()  # Текущее время
            }
            data.append(news_data)

        # Исправленный запрос для SQLite
        stmt = sqlite_insert(News).values(data)
        stmt = stmt.on_conflict_do_nothing(index_elements=['id'])
        await self.session.execute(stmt)

    @handle_db_errors
    async def get_unprocessed_news(self, limit: int = 10):
        stmt = select(News).where(
            News.processed == False
        ).order_by(
            News.date.desc()
        ).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    @handle_db_errors
    async def save_analysis(self, news_id: str, analysis: str):
        stmt = sqlite_insert(Analysis).values(
            news_id=news_id,
            analysis=analysis,
            published=False,
            published_at=None,  # Явное указание NULL
            created_at=datetime.utcnow()
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['news_id'],
            set_={
                "analysis": stmt.excluded.analysis,
                "published": stmt.excluded.published,
                "published_at": stmt.excluded.published_at
            }
        )
        await self.session.execute(stmt)

        stmt = update(News).where(
            News.id == news_id
        ).values(
            processed=True,
            processed_at=datetime.utcnow()
        )
        await self.session.execute(stmt)

    @handle_db_errors
    async def get_unpublished_analysis(self, limit: int = 10):  # Увеличили лимит
        stmt = select(Analysis).join(News).where(
            Analysis.published == False
        ).options(
            selectinload(Analysis.news)
        ).order_by(
            News.date.desc()
        ).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    @handle_db_errors
    async def mark_as_published(self, news_id: str):
        stmt = update(Analysis).where(
            Analysis.news_id == news_id
        ).values(
            published=True,
            published_at=datetime.utcnow()
        )
        await self.session.execute(stmt)