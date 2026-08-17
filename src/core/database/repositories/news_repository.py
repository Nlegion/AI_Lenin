from sqlalchemy import delete, func, select, text, update
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import OperationalError
from src.core.database.models.models import Analysis, CensorDecisionCache, News
from src.core.utils.decorators import handle_db_errors
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from datetime import timedelta
import asyncio
import json
import logging

from src.core.database.utc import utc_now

logger = logging.getLogger(__name__)


class NewsRepository:
    def __init__(self, session):
        self.session = session
        self.stats = {
            "news_fetched": 0,
            "news_processed": 0,
            "analyses_published": 0,
            "errors": 0,
        }

    async def _execute_with_retry(self, statement, *, retries: int = 3):
        delay = 0.05
        for attempt in range(retries + 1):
            try:
                return await self.session.execute(statement)
            except OperationalError as error:
                if "database is locked" not in str(error).lower() or attempt >= retries:
                    raise
                logger.warning(
                    "SQLite lock during query, retrying attempt=%s delay=%.3f",
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2

    @handle_db_errors
    async def mark_as_processed_without_analysis(self, news_id: str):
        """Помечает новость как обработанную без сохранения анализа"""
        stmt = (
            update(News)
            .where(News.id == news_id)
            .values(processed=True, processed_at=utc_now())
        )
        await self.session.execute(stmt)

    @handle_db_errors
    async def save_news(self, news_items: list) -> int:
        """Insert news rows. Returns count of newly inserted rows."""
        if not news_items:
            return 0

        logger.info(f"Сохранение {len(news_items)} новостей в БД")

        data = []
        for item in news_items:
            news_data = {
                "id": item["id"],
                "title": item["title"],
                "content": item["content"],
                "source": item["source"],
                "date": item["date"],
                "url": item["url"],
                "processed": False,
                "processed_at": None,
                "created_at": utc_now(),
            }
            data.append(news_data)

        stmt = sqlite_insert(News).values(data)
        stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
        result = await self._execute_with_retry(stmt)
        inserted = int(getattr(result, "rowcount", None) or 0)
        # SQLite/aiosqlite may report -1 or the attempt count; prefer changes().
        if inserted < 0 or inserted > len(news_items):
            changes = await self.session.execute(text("SELECT changes()"))
            inserted = int(changes.scalar_one() or 0)
        return max(0, min(inserted, len(news_items)))

    @handle_db_errors
    async def get_news_by_id(self, news_id: str):
        stmt = select(News).where(News.id == news_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    @handle_db_errors
    async def get_unprocessed_news(self, limit: int = 10):
        # Фильтруем новости не старше 24 часов
        time_threshold = utc_now() - timedelta(hours=24)

        stmt = (
            select(News)
            .where(
                News.processed.is_(False),
                News.date >= time_threshold,  # Только свежие новости
            )
            .order_by(News.date.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    @handle_db_errors
    async def count_unprocessed(self, *, hours: int = 24) -> int:
        time_threshold = utc_now() - timedelta(hours=hours)
        stmt = (
            select(func.count())
            .select_from(News)
            .where(
                News.processed.is_(False),
                News.date >= time_threshold,
            )
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one() or 0)

    @handle_db_errors
    async def count_stale_unprocessed(self, *, hours: int = 24) -> int:
        time_threshold = utc_now() - timedelta(hours=hours)
        stmt = (
            select(func.count())
            .select_from(News)
            .where(
                News.processed.is_(False),
                News.date < time_threshold,
            )
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one() or 0)

    @handle_db_errors
    async def count_unpublished(self) -> int:
        stmt = (
            select(func.count())
            .select_from(Analysis)
            .where(Analysis.published.is_(False))
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one() or 0)

    @handle_db_errors
    async def save_analysis(self, news_id: str, analysis: str):
        stmt = sqlite_insert(Analysis).values(
            news_id=news_id,
            analysis=analysis,
            published=False,
            published_at=None,  # Явное указание NULL
            created_at=utc_now(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["news_id"],
            set_={
                "analysis": stmt.excluded.analysis,
                "published": stmt.excluded.published,
                "published_at": stmt.excluded.published_at,
            },
        )
        await self._execute_with_retry(stmt)

        stmt = (
            update(News)
            .where(News.id == news_id)
            .values(processed=True, processed_at=utc_now())
        )
        await self._execute_with_retry(stmt)

    @handle_db_errors
    async def get_unpublished_analysis(self, limit: int = 10):  # Увеличили лимит
        stmt = (
            select(Analysis)
            .join(News)
            .where(Analysis.published.is_(False))
            .options(selectinload(Analysis.news))
            .order_by(News.date.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    @handle_db_errors
    async def mark_as_published(self, news_id: str):
        stmt = (
            update(Analysis)
            .where(Analysis.news_id == news_id)
            .values(published=True, published_at=utc_now())
        )
        await self.session.execute(stmt)

    @handle_db_errors
    async def mark_analysis_discarded(self, news_id: str) -> None:
        """Terminal discard: leave unpublished queue without counting as published."""
        stmt = (
            update(Analysis)
            .where(Analysis.news_id == news_id)
            .values(published=True, published_at=utc_now())
        )
        await self.session.execute(stmt)
        news_stmt = (
            update(News)
            .where(News.id == news_id)
            .values(processed=True, processed_at=utc_now())
        )
        await self.session.execute(news_stmt)

    @handle_db_errors
    async def mark_as_unprocessed(self, news_id: str):
        """Помечает новость как необработанную для повторной попытки"""
        stmt = (
            update(News)
            .where(News.id == news_id)
            .values(processed=False, processed_at=None)
        )
        await self._execute_with_retry(stmt)

    @handle_db_errors
    async def get_censor_cached_decision(
        self,
        *,
        content_hash: str,
        config_version_hash: str,
    ) -> dict | None:
        stmt = select(CensorDecisionCache).where(
            CensorDecisionCache.content_hash == content_hash,
            CensorDecisionCache.config_version_hash == config_version_hash,
        )
        result = await self._execute_with_retry(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        touch = (
            update(CensorDecisionCache)
            .where(
                CensorDecisionCache.content_hash == content_hash,
                CensorDecisionCache.config_version_hash == config_version_hash,
            )
            .values(
                last_accessed_at=utc_now(),
                hit_count=CensorDecisionCache.hit_count + 1,
            )
        )
        await self._execute_with_retry(touch)
        confidence = json.loads(row.confidence_json) if row.confidence_json else {}
        context_hints = list(confidence.pop("__context_hints__", []) or [])
        needs_yellow_warning = bool(confidence.pop("__needs_yellow_warning__", False))
        return {
            "decision": row.decision,
            "category": row.category,
            "risk_tier": row.risk_tier,
            "reason_codes": json.loads(row.reason_codes_json),
            "confidence": confidence,
            "context_hints": context_hints,
            "needs_yellow_warning": needs_yellow_warning,
            "model_version_hash": row.model_version_hash,
        }

    @handle_db_errors
    async def upsert_censor_cached_decision(
        self,
        *,
        content_hash: str,
        config_version_hash: str,
        model_version_hash: str,
        decision: str,
        category: str | None,
        risk_tier: str,
        reason_codes: list[str],
        confidence: dict,
    ) -> None:
        now = utc_now()
        stmt = sqlite_insert(CensorDecisionCache).values(
            content_hash=content_hash,
            config_version_hash=config_version_hash,
            model_version_hash=model_version_hash,
            decision=decision,
            category=category,
            risk_tier=risk_tier,
            reason_codes_json=json.dumps(reason_codes, ensure_ascii=False),
            confidence_json=json.dumps(confidence, ensure_ascii=False),
            created_at=now,
            last_accessed_at=now,
            hit_count=0,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["content_hash", "config_version_hash"],
            set_={
                "model_version_hash": stmt.excluded.model_version_hash,
                "decision": stmt.excluded.decision,
                "category": stmt.excluded.category,
                "risk_tier": stmt.excluded.risk_tier,
                "reason_codes_json": stmt.excluded.reason_codes_json,
                "confidence_json": stmt.excluded.confidence_json,
                "last_accessed_at": stmt.excluded.last_accessed_at,
            },
        )
        await self._execute_with_retry(stmt)

    @handle_db_errors
    async def cleanup_censor_cache(self, *, max_age_seconds: int) -> int:
        cutoff = utc_now() - timedelta(seconds=max_age_seconds)
        stmt = delete(CensorDecisionCache).where(
            CensorDecisionCache.last_accessed_at < cutoff
        )
        result = await self._execute_with_retry(stmt)
        return int(result.rowcount or 0)
