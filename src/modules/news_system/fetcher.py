import feedparser
import requests
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

    def fetch_rss(self, url: str) -> list:
        try:
            feed = feedparser.parse(url)
            return [{
                "id": self._generate_id(entry.link),
                "title": entry.title,
                "content": entry.get('description', entry.title),  # Fallback на заголовок
                "source": f"RSS: {url}",
                "date": datetime(*entry.published_parsed[:6]) if hasattr(entry,
                                                                         'published_parsed') else datetime.utcnow(),
                "url": entry.link
            } for entry in feed.entries if entry.link]
        except Exception as e:
            logger.error(f"Ошибка RSS {url}: {str(e)}")
            return []

    def fetch_newsapi(self) -> list:
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                'apiKey': self.config.NEWSAPI_KEY,
                'country': 'ru',
                'language': 'ru',
                'pageSize': 50
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            articles = response.json().get('articles', [])
            russian_articles = []

            for article in articles:
                if article.get('language', 'ru') != 'ru':
                    continue

                if not all([article.get('title'), article.get('url')]):
                    continue

                try:
                    pub_date = article['publishedAt'].replace('Z', '+00:00')
                    date = datetime.fromisoformat(pub_date)
                except:
                    date = datetime.utcnow()

                russian_articles.append({
                    "id": self._generate_id(article['url']),
                    "title": article['title'],
                    "content": article.get('description', article['title']),
                    "source": f"NewsAPI: {article['source']['name']}",
                    "date": date,
                    "url": article['url']
                })

            return russian_articles
        except Exception as e:
            logger.error(f"Ошибка NewsAPI: {str(e)}")
            return []

    def fetch_all(self) -> list:
        news_items = []

        # RSS источники
        rss_sources = [
            "https://lenta.ru/rss/news",
            "https://www.interfax.ru/rss.asp",
            "https://ria.ru/export/rss2/archive/index.xml",
            "https://tass.ru/rss/v2.xml",
        ]

        for url in rss_sources:
            news_items.extend(self.fetch_rss(url))

        # Новости из NewsAPI
        news_items.extend(self.fetch_newsapi())

        # Фильтрация по языку (кириллица)
        filtered_items = []
        for item in news_items:
            text = f"{item['title']} {item['content']}"
            if any('\u0400' <= char <= '\u04FF' for char in text):
                filtered_items.append(item)

        # Фильтрация по дате (последние 3 дня)
        return [
            item for item in filtered_items
            if datetime.now() - item['date'] < timedelta(days=3)
        ]