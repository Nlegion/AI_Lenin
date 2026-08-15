import logging
from src.core.adapters.telegram.service import TelegramService
from src.core.settings.config import Settings
from src.core.version import version_manager
import re

logger = logging.getLogger(__name__)


def clean_telegram_text(text: str) -> str:
    """Удаляет проблемные символы для Telegram"""
    # Удаляем непечатаемые символы
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
    # Заменяем проблемные HTML-сущности
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.strip()


class TelegramPublisher:
    def __init__(self):
        self.config = Settings()
        self.service = TelegramService()
        self.version = version_manager.get_full_version()

    async def publish_analysis(
        self, news_id: str, title: str, url: str, analysis: str
    ) -> bool:
        try:
            logger.info(f"Публикация: {title[:30]}...")

            # Экранирование HTML-символов
            title = clean_telegram_text(title)
            analysis = clean_telegram_text(analysis)

            # Reserve room for footer/disclaimer + framing (Telegram hard limit 4096).
            telegram_limit = 4096
            framing_overhead = (
                len(f"<b>📰 {title}</b>\n\n")
                + len(f"<i>Модель Ай_Ленин {self.version} 💬</i>\n")
                + len(f"\n\n<a href='{url}'>Источник</a>")
                + 8
            )
            max_analysis = max(200, telegram_limit - framing_overhead)
            if len(analysis) > max_analysis:
                logger.warning(
                    "Слишком длинное сообщение, сокращаем тело (footer preserved)"
                )
                # Prefer keeping a trailing short disclaimer paragraph.
                parts = analysis.rsplit("\n\n", 1)
                if len(parts) == 2 and len(parts[1]) < 280:
                    body, footer = parts
                    budget = max(80, max_analysis - len(footer) - 4)
                    analysis = f"{body[:budget].rstrip()}...\n\n{footer}"
                else:
                    analysis = analysis[: max_analysis - 3].rstrip() + "..."

            message = (
                f"<b>📰 {title}</b>\n\n"
                f"<i>Модель Ай_Ленин {self.version} 💬</i>\n"
                f"{analysis}\n\n"
                f"<a href='{url}'>Источник</a>"
            )

            # Отправка сообщения
            response = await self.service.send_message(
                chat_id=self.config.TELEGRAM_CHANNEL_ID,
                text=message,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )

            # Проверка ответа
            if response and response.get("ok", False):
                return True

            logger.error(f"Ошибка публикации: {response}")
            return False

        except Exception as e:
            logger.exception(f"Ошибка публикации: {str(e)}")
            return False

    async def send_admin_notification(self, message: str) -> bool:
        try:
            if not self.config.TELEGRAM_ADMIN_ID:
                return False

            response = await self.service.send_message(
                chat_id=self.config.TELEGRAM_ADMIN_ID, text=message
            )
            return response.get("ok", False)
        except Exception as e:
            logger.error(f"Ошибка уведомления админа: {str(e)}")
            return False
