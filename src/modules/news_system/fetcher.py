import feedparser
import requests
import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from src.core.settings.config import Settings
from telethon import TelegramClient

logger = logging.getLogger(__name__)


class NewsFetcher:
    def __init__(self):
        self.config = Settings()

    def _generate_id(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def fetch_rss(self, url: str) -> list:
        try:
            feed = feedparser.parse(url)
            return [{
                "id": self._generate_id(entry.link),
                "title": entry.title,
                "content": entry.description,
                "source": f"RSS: {url}",
                "date": datetime(*entry.published_parsed[:6]),
                "url": entry.link
            } for entry in feed.entries if hasattr(entry, 'published_parsed')]
        except Exception as e:
            logger.error(f"Ошибка RSS {url}: {str(e)}")
            return []

    def fetch_newsapi(self) -> list:
        try:
            url = "https://newsapi.org/v2/top-headlines"
            response = requests.get(url, params={
                'apiKey': self.config.NEWSAPI_KEY,
                'sources': 'bbc-news,reuters',
                'pageSize': 20
            }, timeout=10)

            return [{
                "id": self._generate_id(article['url']),
                "title": article['title'],
                "content": article['description'],
                "source": f"NewsAPI: {article['source']['name']}",
                "date": datetime.fromisoformat(article['publishedAt'][:-1]),
                "url": article['url']
            } for article in response.json().get('articles', [])]
        except Exception as e:
            logger.error(f"Ошибка NewsAPI: {str(e)}")
            return []

    async def _fetch_telegram_channel(self, channel: str) -> list:
        try:
            client = TelegramClient('session', self.config.TELEGRAM_API_ID, self.config.TELEGRAM_API_HASH)
            await client.start()

            return [{
                "id": self._generate_id(f"{channel}_{msg.id}"),
                "title": msg.text[:100],
                "content": msg.text,
                "source": f"Telegram: {channel}",
                "date": msg.date,
                "url": f"https://t.me/{channel}/{msg.id}"
            } async for msg in client.iter_messages(channel, limit=50) if msg.text]
        except Exception as e:
            logger.error(f"Ошибка Telegram {channel}: {str(e)}")
            return []

    def fetch_all(self) -> list:
        news_items = []

        # RSS
        for url in ["https://lenta.ru/rss/news", "https://www.interfax.ru/rss.asp"]:
            news_items.extend(self.fetch_rss(url))

        # NewsAPI
        news_items.extend(self.fetch_newsapi())

        # Telegram
        for channel in ["rian_ru", "tass_agency"]:
            news_items.extend(asyncio.run(self._fetch_telegram_channel(channel)))

        # Фильтрация по дате
        return [item for item in news_items if datetime.now() - item['date'] < timedelta(days=3)]