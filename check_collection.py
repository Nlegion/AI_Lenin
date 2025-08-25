import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path


def check_collection_stats():
    """Проверка статистики коллекции"""

    # Путь к векторной БД
    vector_db_path = Path(r"P:\AI_Lenin\database\rag_db")

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

        # Получение примеров документов
        if count > 0:
            sample = collection.peek(limit=5)
            print("\nПримеры документов:")
            for i, (doc, metadata) in enumerate(zip(sample['documents'], sample['metadatas'])):
                print(f"{i + 1}. Автор: {metadata['author']}, Работа: {metadata['work']}")
                print(f"   Текст: {doc[:100]}...")
                print()

    except Exception as e:
        print(f"Ошибка при получении коллекции: {e}")


if __name__ == "__main__":
    check_collection_stats()