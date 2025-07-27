import logging
from src.core.adapters.telegram.service import TelegramService
from src.core.settings.config import Settings
import html

logger = logging.getLogger(__name__)


class TelegramPublisher:
    def __init__(self):
        self.config = Settings()
        self.service = TelegramService()

    async def publish_analysis(self, news_id: str, title: str, url: str, analysis: str) -> bool:
        try:
            logger.info(f"Публикация: {title[:30]}...")

            message = (
                f"<b>📰 {html.escape(title)}</b>\n\n"
                f"<i>💬 Анализ Владимира Ильича Ленина:</i>\n"
                f"{analysis}\n\n"
                f"<a href='{html.escape(url)}'>Источник</a>"
            )

            response = await self.service.send_message(
                chat_id=self.config.TELEGRAM_CHANNEL_ID,
                text=message,
                parse_mode="HTML",
                disable_web_page_preview=True
            )

            return response.get("ok", False)
        except Exception as e:
            logger.error(f"Ошибка публикации: {str(e)}")
            return False

    async def send_admin_notification(self, message: str) -> bool:
        try:
            if not self.config.TELEGRAM_ADMIN_ID:
                return False

            response = await self.service.send_message(
                chat_id=self.config.TELEGRAM_ADMIN_ID,
                text=message
            )
            return response.get("ok", False)
        except Exception as e:
            logger.error(f"Ошибка уведомления админа: {str(e)}")
            return False