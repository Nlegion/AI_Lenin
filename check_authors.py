import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path


def check_authors():
    """Проверка авторов в коллекции"""

    # Путь к векторной БД
    vector_db_path = Path(r"P:\AI_Lenin\database\rag_db")

    if not vector_db_path.exists():
        print("Векторная БД не существует! Сначала постройте индекс.")
        return

    # Инициализация клиента
    client = chromadb.PersistentClient(path=str(vector_db_path))

    # Функция для эмбеддингов
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        # Получение коллекции
        collection = client.get_collection(
            name="philosophy_ontology",
            embedding_function=embedding_function
        )

        # Получение статистики
        count = collection.count()

        print("=" * 60)
        print("СТАТИСТИКА КОЛЛЕКЦИИ")
        print("=" * 60)
        print(f"Количество документов: {count}")

        # Получение уникальных авторов
        sample_size = min(5000, count)
        sample = collection.peek(limit=sample_size)
        authors = set()

        for metadata in sample['metadatas']:
            authors.add(metadata['author'])

        print(f"\nУникальные авторы в коллекции: {len(authors)}")
        for author in sorted(authors):
            try:
                # Используем правильный синтаксис для запроса
                results = collection.get(
                    where={"author": {"$eq": author}},
                    limit=1
                )
                author_count = len(results['ids'])
                print(f"  - {author}: {author_count} документов")
            except Exception as e:
                print(f"  - {author}: ошибка подсчета - {e}")

        # Проверяем наличие ключевых авторов
        key_authors = ["Ленин", "МарксЭнгельс", "Гегель", "Аристотель", "Богданов", "Плеханов"]
        print("\nНаличие ключевых авторов:")
        for author in key_authors:
            try:
                results = collection.get(
                    where={"author": {"$eq": author}},
                    limit=1
                )
                author_count = len(results['ids'])
                print(f"  - {author}: {author_count} документов")
            except:
                print(f"  - {author}: не найден")

        # Показываем примеры документов
        print("\nПримеры документов:")
        sample = collection.peek(limit=5)
        for i, (doc, metadata) in enumerate(zip(sample['documents'], sample['metadatas'])):
            print(f"{i + 1}. Автор: {metadata['author']}, Работа: {metadata['work']}")
            print(f"   Текст: {doc[:200]}...")
            print()

    except Exception as e:
        print(f"Ошибка при получении коллекции: {e}")


if __name__ == "__main__":
    check_authors()