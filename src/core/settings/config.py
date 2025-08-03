import os
import torch
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Модель ИИ
    EMBEDDING_MODEL = "models/multilingual-e5-large"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
    TELEGRAM_ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID")

    # База данных
    DB_PATH = "ai_lenin.db"

    # Параметры генерации
    MODEL_PATH = "models/deepseek/deepseek-r1-7b.Q4_K_M.gguf"
    MAX_TOKENS = 256
    TEMPERATURE = 0.8
    TOP_P = 0.95
    UPDATE_INTERVAL = 7200
    MAX_NEWS_PER_CYCLE = 5

    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_DIR = os.path.join(BASE_DIR, "models", "saiga")
    VECTOR_DB_PATH = os.path.join(BASE_DIR, "database", "vector_db")