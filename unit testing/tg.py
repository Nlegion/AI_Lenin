import requests
import os
from dotenv import load_dotenv

load_dotenv()

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
channel_username = "@ai_lenin_news_chat"  # Например: "@lenin_news"

# Получить ID канала
response = requests.get(
    f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={channel_username}&text=Test"
)
print(response.json())