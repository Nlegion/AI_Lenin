import requests

bot_token = "8267003872:AAH083D6OXb_Me8ho4MrjJ70aMqJ9NQTGfk"
channel_username = "@ai_lenin_news_chat"  # Например: "@lenin_news"

# Получить ID канала
response = requests.get(
    f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={channel_username}&text=Test"
)
print(response.json())