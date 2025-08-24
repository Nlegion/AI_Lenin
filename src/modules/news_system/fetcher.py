import feedparser
import hashlib
import logging
from datetime import datetime, timedelta
from src.core.settings.config import Settings

logger = logging.getLogger(__name__)


class NewsFetcher:
    def __init__(self):
        self.config = Settings()

    def _generate_id(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def fetch_tass(self) -> list:
        """Сбор новостей только из TASS"""
        try:
            # Основной RSS TASS
            feed = feedparser.parse("https://tass.ru/rss/v2.xml")

            return [{
                "id": self._generate_id(entry.link),
                "title": entry.title,
                "content": entry.get('description', entry.title),
                "source": "TASS",
                "date": datetime(*entry.published_parsed[:6]) if hasattr(entry,
                                                                         'published_parsed') else datetime.utcnow(),
                "url": entry.link
            } for entry in feed.entries if entry.link]
        except Exception as e:
            logger.error(f"Ошибка TASS RSS: {str(e)}")
            return []

    def fetch_all(self) -> list:
        """Сбор новостей только из TASS"""
        logger.info("Сбор новостей из TASS")
        return self.fetch_tass()