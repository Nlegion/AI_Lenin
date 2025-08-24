import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.modules.news_system.fetcher import NewsFetcher
from src.core.settings.log import setup_logging

# Настройка логирования
setup_logging()
logger = logging.getLogger(__name__)


def print_news(news_item):
    """Печатает информацию о новости в читаемом формате"""
    print("\n" + "=" * 80)
    print(f"Заголовок: {news_item['title']}")
    print(f"Источник: {news_item['source']}")
    print(f"Дата: {news_item['date'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"URL: {news_item['url']}")

    # Обрезаем длинный текст для удобства просмотра
    content = news_item['content']
    if len(content) > 300:
        content = content[:300] + "... [truncated]"

    print("\nСодержание:")
    print(content)
    print("=" * 80)


async def test_fetcher():
    logger.info("Запуск теста модуля сбора новостей")

    # Проверка наличия ключа NewsAPI
    if not os.getenv("NEWSAPI_KEY"):
        logger.warning("NEWSAPI_KEY не установлен. Будут использоваться только RSS источники")

    # Создаем экземпляр сборщика новостей
    fetcher = NewsFetcher()

    # Получаем новости
    logger.info("Начало сбора новостей...")
    start_time = datetime.now()
    news_items = fetcher.fetch_all()
    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info(f"Сбор завершен. Получено новостей: {len(news_items)}")
    logger.info(f"Время выполнения: {elapsed:.2f} секунд")

    if not news_items:
        logger.error("Не удалось получить новости. Проверьте подключение к интернету и настройки.")
        return

    # Фильтруем новости по источнику для анализа
    sources = {}
    for item in news_items:
        sources[item['source']] = sources.get(item['source'], 0) + 1

    # Выводим статистику
    print("\nСтатистика по источникам:")
    for source, count in sources.items():
        print(f"- {source}: {count} новостей")

    # Проверяем временной диапазон
    now = datetime.now()
    oldest = min(item['date'] for item in news_items)
    newest = max(item['date'] for item in news_items)

    print(f"\nВременной диапазон новостей:")
    print(f"- Самая старая: {oldest.strftime('%Y-%m-%d %H:%M')}")
    print(f"- Самая новая: {newest.strftime('%Y-%m-%d %H:%M')}")
    print(f"- Разница: {(now - oldest).days} дней назад")

    # Проверяем язык новостей
    non_russian = 0
    for item in news_items:
        text = f"{item['title']} {item['content']}"
        if not any('\u0400' <= char <= '\u04FF' for char in text):
            non_russian += 1
            print(f"\n⚠️ Обнаружена не русскоязычная новость:")
            print(f"Заголовок: {item['title']}")
            print(f"Источник: {item['source']}")

    if non_russian:
        logger.warning(f"Найдено {non_russian} не русскоязычных новостей")
    else:
        logger.info("Все новости на русском языке")

    # Выводим примеры новостей
    print("\nПримеры новостей:")
    for i, item in enumerate(news_items[:3]):
        print_news(item)

    # Сохраняем результаты в файл
    output_file = "fetcher_test_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Тест модуля сбора новостей ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n")
        f.write(f"Всего новостей: {len(news_items)}\n\n")

        for i, item in enumerate(news_items):
            f.write(f"Новость #{i + 1}\n")
            f.write(f"Источник: {item['source']}\n")
            f.write(f"Дата: {item['date'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"URL: {item['url']}\n")
            f.write(f"Заголовок: {item['title']}\n")
            f.write(f"Содержание:\n{item['content']}\n")
            f.write("-" * 80 + "\n")

    logger.info(f"Полные результаты сохранены в файл: {output_file}")


if __name__ == "__main__":
    asyncio.run(test_fetcher())