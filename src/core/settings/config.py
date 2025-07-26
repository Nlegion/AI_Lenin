import os
import torch
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Модель ИИ
    MODEL_NAME = "IlyaGusev/saiga_mistral_7b"
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
    TELEGRAM_ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID")
    TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
    TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

    # База данных
    DB_PATH = "ai_lenin.db"

    # Параметры генерации
    MAX_TOKENS = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    UPDATE_INTERVAL = 3600  # Интервал обработки в секундах