import asyncio
import logging
from pathlib import Path
from src.core.rag_system import EnhancedRAGSystem

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_rag_system():
    """Тестирование RAG системы с различными запросами"""

    # Инициализация RAG системы
    ontology_path = r"P:\AI_Lenin\data\books\ultimate_cleaned_ontology"
    rag_system = EnhancedRAGSystem(ontology_path=ontology_path)

    # Тестовые запросы для проверки работы системы
    test_queries = [
        "капитализм и эксплуатация",
        "диалектический материализм",
        "классовая борьба",
        "революция пролетариата",
        "прибавочная стоимость",
        "государство и революция",
        "империализм как высшая стадия капитализма",
        "критика идеализма",
        "роль партии в революции",
        "кооперация и коллективизация"
    ]

    print("=" * 80)
    print("ТЕСТИРОВАНИЕ RAG СИСТЕМЫ С ОЧИЩЕННОЙ ОНТОЛОГИЕЙ")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. ЗАПРОС: {query}")
        print("-" * 60)

        # Поиск только у Ленина
        print("Результаты от Ленина:")
        context_lenin = rag_system.retrieve_relevant_context(
            query,
            k=3,
            author_filter="Ленин"
        )

        if context_lenin:
            # Показываем первые 2 результата полностью или обрезаем
            contexts = context_lenin.split('\n\n')
            for j, context in enumerate(contexts[:2], 1):
                print(f"{j}. {context}")
                if j < len(contexts[:2]):
                    print()
        else:
            print("Не найдено результатов от Ленина")

        # Поиск у всех авторов
        print("\nРезультаты от всех авторов:")
        context_all = rag_system.retrieve_relevant_context(query, k=3)

        if context_all:
            contexts = context_all.split('\n\n')
            for j, context in enumerate(contexts[:2], 1):
                print(f"{j}. {context}")
                if j < len(contexts[:2]):
                    print()
        else:
            print("Не найдено результатов")

        print("-" * 60)
        await asyncio.sleep(1)  # Пауза между запросами


async def test_specific_queries():
    """Тестирование конкретных запросов с глубоким анализом"""

    ontology_path = r"P:\AI_Lenin\data\books\ultimate_cleaned_ontology"
    rag_system = EnhancedRAGSystem(ontology_path=ontology_path)

    deep_queries = [
        "прибавочная стоимость",
        "диалектический материализм",
        "империализм"
    ]

    print("\n" + "=" * 80)
    print("ГЛУБОКОЕ ТЕСТИРОВАНИЕ КЛЮЧЕВЫХ ЗАПРОСОВ")
    print("=" * 80)

    for query in deep_queries:
        print(f"\nЗАПРОС: {query}")
        print("-" * 40)

        # Ищем у разных авторов
        authors = ["Ленин", "МарксЭнгельс", "Гегель"]

        for author in authors:
            print(f"\nРезультаты от {author}:")
            context = rag_system.retrieve_relevant_context(
                query,
                k=2,
                author_filter=author
            )

            if context:
                print(context)
            else:
                print("Не найдено результатов")

        print("-" * 40)


async def main():
    """Основная функция тестирования"""

    print("Запуск тестирования RAG системы с очищенной онтологией...")

    # Тест 1: Общее тестирование
    await test_rag_system()

    # Тест 2: Глубокое тестирование ключевых запросов
    await test_specific_queries()

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())