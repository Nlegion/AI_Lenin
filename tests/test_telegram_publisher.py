import asyncio
import logging
import os
from dotenv import load_dotenv
from src.core.publisher import TelegramPublisher
from src.core.settings.log import setup_logging

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
setup_logging()
logger = logging.getLogger(__name__)


async def test_telegram_publisher():
    logger.info("Запуск теста Telegram Publisher")

    # Создаем экземпляр издателя
    publisher = TelegramPublisher()

    # Тест 1: Отправка администратору
    admin_id = os.getenv("TELEGRAM_ADMIN_ID")
    if admin_id:
        logger.info(f"Отправка тестового сообщения администратору (ID: {admin_id})")
        success = await publisher.send_admin_notification("🔧 Тестовое сообщение от системы ИИ-Ленин\n"
                                                          "Это сообщение подтверждает работоспособность модуля публикации!")
        logger.info(f"Результат отправки администратору: {'Успешно' if success else 'Ошибка'}")
    else:
        logger.warning("TELEGRAM_ADMIN_ID не установлен, пропускаем тест администратора")

    # Тест 2: Публикация в канал
    channel_id = os.getenv("TELEGRAM_CHANNEL_ID")
    if channel_id:
        logger.info(f"Публикация тестового сообщения в канал (ID: {channel_id})")
        success = await publisher.publish_analysis(
            news_id="test_123",
            title="Тестовая новость: Работоспособность системы",
            url="https://example.com",
            analysis="✅ Система успешно отправляет сообщения!\n\n"
                     "Владимир Ильич Ленин одобряет исправную работу революционной техники. "
                     "Пролетарии всех стран, соединяйтесь в цифровом пространстве!"
        )
        logger.info(f"Результат публикации в канал: {'Успешно' if success else 'Ошибка'}")
    else:
        logger.warning("TELEGRAM_CHANNEL_ID не установлен, пропускаем тест канала")

    logger.info("Тестирование завершено")


if __name__ == "__main__":
    asyncio.run(test_telegram_publisher())