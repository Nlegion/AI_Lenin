import os
import torch
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()


class Settings:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
    TELEGRAM_ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID")
    # VPS-only SOCKS exit for Bot API (do not set global HTTP(S)_PROXY)
    TELEGRAM_PROXY_URL = os.getenv("TELEGRAM_PROXY_URL")
    TELEGRAM_PROXY_REQUIRED = os.getenv("TELEGRAM_PROXY_REQUIRED", "false")

    # База данных
    DB_PATH = "ai_lenin.db"

    # Параметры цикла новостей
    MAX_TOKENS = 256
    TEMPERATURE = 0.8
    TOP_P = 0.95
    UPDATE_INTERVAL = 300
    MAX_NEWS_PER_CYCLE = 3

    # TASS RSS: StormWall returns HTTP 403 for feedparser's default User-Agent.
    TASS_RSS_URL = "https://tass.ru/rss/v2.xml"
    TASS_RSS_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    )
    TASS_RSS_ACCEPT = "application/rss+xml, application/xml, text/xml, */*"

    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

    # settings/config.py -> settings -> core -> src -> repo root
    BASE_DIR = str(Path(__file__).resolve().parents[3])

    ontology_path: str = str(Path(BASE_DIR) / "data" / "books" / "intellectual")

    class Config:
        env_file = ".env"
