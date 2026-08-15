import logging
import re
from pathlib import Path

from src.core.adapters.telegram.service import TelegramService
from src.core.settings.config import Settings
from src.core.settings.quality_postcheck_config import (
    default_quality_postcheck_path,
    get_quality_postcheck_config,
)

logger = logging.getLogger(__name__)


def _load_ai_disclaimer() -> str:
    """Public Telegram footer SoT: quality_postcheck.short_disclaimer."""
    # src/core/publisher.py -> core -> src -> repo root
    root = Path(__file__).resolve().parents[2]
    path = default_quality_postcheck_path(base_dir=root)
    fallback = (
        "Ответ сгенерирован ИИ в образовательных целях "
        "(симуляция на основе трудов В.И. Ленина) и не является призывом к действию."
    )
    if not path.is_file():
        return fallback
    cfg = get_quality_postcheck_config(path_str=str(path))
    return (cfg.short_disclaimer or "").strip() or fallback


AI_DISCLAIMER = _load_ai_disclaimer()

_TRIAD_LINE = re.compile(
    r"(?im)^\s*\*{0,2}(факт|механизм|вывод)\*{0,2}\s*:\s*(.*?)(?=\n|$)"
)


def clean_telegram_text(text: str) -> str:
    """Удаляет проблемные символы для Telegram"""
    # Удаляем непечатаемые символы
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
    # Заменяем проблемные HTML-сущности
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.strip()


def _extract_triad_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for match in _TRIAD_LINE.finditer(text):
        label = match.group(1).casefold()
        value = match.group(2).strip()
        if value and label not in sections:
            sections[label] = value
    return sections


class TelegramPublisher:
    def __init__(self):
        self.config = Settings()
        self.service = TelegramService()

    async def publish_analysis(
        self, news_id: str, title: str, url: str, analysis: str
    ) -> bool:
        try:
            logger.info(f"Публикация: {title[:30]}...")

            # Экранирование HTML-символов
            title = clean_telegram_text(title)
            analysis = clean_telegram_text(analysis)

            triad_sections = _extract_triad_sections(analysis)
            mechanism = triad_sections.get("механизм", "").strip()
            conclusion = triad_sections.get("вывод", "").strip()
            if not mechanism or not conclusion:
                logger.warning(
                    "Публикация отменена: отсутствуют обязательные секции triad",
                    extra={
                        "has_mechanism": bool(mechanism),
                        "has_conclusion": bool(conclusion),
                        "news_id": news_id,
                    },
                )
                return False

            # Reserve room for body + source + disclaimer (Telegram limit 4096).
            telegram_limit = 4096
            framing_overhead = (
                len(f"<b>Факт: {title}</b>\n\n")
                + len("Механизм: \n\n")
                + len("Вывод: \n\n")
                + len(f"<a href='{url}'>источник</a>\n\n")
                + len(AI_DISCLAIMER)
                + 48
            )
            max_sections = max(200, telegram_limit - framing_overhead)
            if len(mechanism) + len(conclusion) > max_sections:
                logger.warning("Слишком длинное сообщение, сокращаем sections")
                mechanism_budget = max(100, int(max_sections * 0.6))
                conclusion_budget = max(80, max_sections - mechanism_budget)
                mechanism = mechanism[:mechanism_budget].rstrip()
                conclusion = conclusion[:conclusion_budget].rstrip()
                if len(mechanism) == mechanism_budget:
                    mechanism = mechanism.rstrip(". ") + "..."
                if len(conclusion) == conclusion_budget:
                    conclusion = conclusion.rstrip(". ") + "..."

            message = (
                f"<b>Факт: {title}</b>\n\n"
                f"Механизм: {mechanism}\n\n"
                f"Вывод: {conclusion}\n\n"
                f"<a href='{url}'>источник</a>\n\n"
                f"{AI_DISCLAIMER}"
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
