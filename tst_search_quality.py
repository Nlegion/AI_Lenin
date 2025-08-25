import asyncio
from src.core.rag_system import EnhancedRAGSystem


async def test_search_quality():
    """Тестирование качества поиска по конкретным терминам"""

    ontology_path = r"P:\AI_Lenin\data\books\intellectual"
    rag_system = EnhancedRAGSystem(ontology_path=ontology_path)

    # Тестовые пары: запрос - ожидаемые ключевые слова в ответе
    test_cases = [
        {
            "query": "прибавочная стоимость",
            "expected_keywords": ["прибавочная стоимость", "капитал", "Маркс"],
            "author": "МарксЭнгельс"
        },
        {
            "query": "диалектический материализм",
            "expected_keywords": ["диалектика", "материализм", "Гегель"],
            "author": "Ленин"
        },
        {
            "query": "империализм",
            "expected_keywords": ["империализм", "капитализм", "монополии"],
            "author": "Ленин"
        },
        {
            "query": "классовая борьба",
            "expected_keywords": ["класс", "борьба", "пролетариат"],
            "author": "Ленин"
        },
        {
            "query": "революция пролетариата",
            "expected_keywords": ["революция", "пролетариат", "социализм"],
            "author": "Ленин"
        }
    ]

    print("=" * 80)
    print("ТЕСТИРОВАНИЕ КАЧЕСТВА ПОИСКА")
    print("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. ТЕСТ: {test_case['query']}")
        print(f"   Ожидаемые ключевые слова: {', '.join(test_case['expected_keywords'])}")

        # Выполняем поиск
        context = rag_system.retrieve_relevant_context(
            test_case['query'],
            k=3,
            author_filter=test_case['author']
        )

        # Проверяем результаты
        if context:
            # Проверяем наличие ожидаемых ключевых слов
            found_keywords = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in context.lower():
                    found_keywords.append(keyword)

            print(f"   Найдено ключевых слов: {', '.join(found_keywords)}")
            print(f"   Полнота: {len(found_keywords)}/{len(test_case['expected_keywords'])}")

            # Выводим краткий обзор результатов
            print("   Пример результата:")
            lines = context.split('\n\n')
            if lines:
                # Берем первую строку и обрезаем до 150 символов
                sample = lines[0][:150] + "..." if len(lines[0]) > 150 else lines[0]
                print(f"   - {sample}")
        else:
            print("   Результаты не найдены")

        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(test_search_quality())