import hashlib
import logging
from datetime import datetime

import feedparser

from src.core.database.utc import utc_now
from src.core.settings.config import Settings

logger = logging.getLogger(__name__)


class NewsFetcher:
    def __init__(self):
        self.config = Settings()

    def _generate_id(self, url: str) -> str:
        return hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()

    def _tass_request_headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.config.TASS_RSS_USER_AGENT,
            "Accept": self.config.TASS_RSS_ACCEPT,
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        }

    def fetch_tass(self) -> list:
        """Сбор новостей только из TASS"""
        try:
            feed = feedparser.parse(
                self.config.TASS_RSS_URL,
                request_headers=self._tass_request_headers(),
            )
            status = getattr(feed, "status", None)
            bozo = bool(getattr(feed, "bozo", False))
            entries = list(getattr(feed, "entries", []) or [])
            if status is not None and int(status) >= 400:
                logger.error(
                    "tass_rss_http_error status=%s bozo=%s entries=%s",
                    status,
                    bozo,
                    len(entries),
                )
                return []
            if bozo and not entries:
                logger.error(
                    "tass_rss_parse_failed error=%s",
                    getattr(feed, "bozo_exception", None),
                )
                return []

            return [
                {
                    "id": self._generate_id(entry.link),
                    "title": entry.title,
                    "content": entry.get("description", entry.title),
                    "source": "TASS",
                    "date": datetime(*entry.published_parsed[:6])
                    if hasattr(entry, "published_parsed") and entry.published_parsed
                    else utc_now(),
                    "url": entry.link,
                }
                for entry in entries
                if getattr(entry, "link", None)
            ]
        except Exception as error:
            logger.exception("tass_rss_fetch_failed error=%s", error)
            return []

    def fetch_all(self) -> list:
        """Сбор новостей только из TASS"""
        logger.info("Сбор новостей из TASS")
        return self.fetch_tass()
