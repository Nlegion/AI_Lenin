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

    # База данных
    DB_PATH = "ai_lenin.db"

    # Параметры цикла новостей
    MAX_TOKENS = 256
    TEMPERATURE = 0.8
    TOP_P = 0.95
    UPDATE_INTERVAL = 300
    MAX_NEWS_PER_CYCLE = 3

    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

    # settings/config.py -> settings -> core -> src -> repo root
    BASE_DIR = str(Path(__file__).resolve().parents[3])

    ontology_path: str = str(Path(BASE_DIR) / "data" / "books" / "intellectual")

    class Config:
        env_file = ".env"
