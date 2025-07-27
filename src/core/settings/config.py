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
    MODEL_NAME = "models/saiga2_7b_lora"
    MAX_TOKENS = 256
    TEMPERATURE = 0.8
    TOP_P = 0.95
    UPDATE_INTERVAL = 7200

    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")