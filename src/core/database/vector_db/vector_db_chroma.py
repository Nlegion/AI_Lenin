import chromadb
import json
from chromadb.utils import embedding_functions
from tqdm import tqdm
import logging
import os
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Правильно определяем корневую директорию проекта
# Поднимаемся на 4 уровня вверх от текущего файла
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'lenin_chunks.jsonl')
CHROMA_PATH = os.path.join(PROJECT_ROOT, 'database', 'vector_db')


def create_chroma_db():
    logger.info("Начало создания векторной базы ChromaDB")
    logger.info(f"Корневая директория проекта: {PROJECT_ROOT}")
    logger.info(f"Исходный файл: {INPUT_FILE}")
    logger.info(f"Путь к ChromaDB: {CHROMA_PATH}")

    # Проверяем существование файла
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Файл не найден: {INPUT_FILE}")
        logger.info("Убедитесь, что вы выполнили скрипт data_preprocessing.py")
        return

    # Создаем директорию для ChromaDB, если не существует
    os.makedirs(CHROMA_PATH, exist_ok=True)

    # Используем встроенную функцию для Sentence Transformers
    try:
        # Проверяем доступность CUDA
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используемое устройство: {device}")

        # Создаем функцию эмбеддингов
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="intfloat/multilingual-e5-large",
            device=device
        )
    except ImportError as e:
        logger.error(f"Не удалось инициализировать модель Sentence Transformers: {str(e)}")
        return

    # Создаем клиент ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        # Пытаемся удалить старую коллекцию, если существует
        client.delete_collection(name="lenin_works")
        logger.info("Старая коллекция 'lenin_works' удалена")
    except:
        logger.info("Коллекция 'lenin_works' не существует, создаем новую")

    # Создаем новую коллекцию
    collection = client.create_collection(
        name="lenin_works",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info("Коллекция 'lenin_works' успешно создана")

    # Загружаем данные
    documents = []
    metadatas = []
    ids = []

    try:
        # Подсчет строк в файле
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        logger.info(f"Начало загрузки {total_lines} документов из файла")
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Чтение файла"):
                data = json.loads(line)
                documents.append(data['text'])
                metadatas.append(data['metadata'])
                ids.append(data['metadata']['doc_id'])

        logger.info(f"Успешно загружено {len(ids)} документов")
    except Exception as e:
        logger.error(f"Ошибка чтения файла: {str(e)}")
        return

    if not ids:
        logger.error("Нет данных для загрузки в ChromaDB")
        return

    # Пакетная вставка
    batch_size = 1000  # Большой размер пакета для ускорения
    inserted_count = 0

    logger.info(f"Начало добавления {len(ids)} документов в ChromaDB")

    # Создаем прогресс-бар
    progress_bar = tqdm(
        total=len(ids),
        desc="Добавление документов",
        unit="doc"
    )

    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        batch_ids = ids[i:end_idx]
        batch_docs = documents[i:end_idx]
        batch_meta = metadatas[i:end_idx]

        try:
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta
            )
            inserted = len(batch_ids)
            inserted_count += inserted
            progress_bar.update(inserted)
        except Exception as e:
            logger.error(f"Ошибка при добавлении пакета {i}-{end_idx}: {str(e)}")
            # Пробуем добавить по одному
            for j in range(i, end_idx):
                try:
                    collection.add(
                        ids=[ids[j]],
                        documents=[documents[j]],
                        metadatas=[metadatas[j]]
                    )
                    inserted_count += 1
                    progress_bar.update(1)
                except:
                    logger.error(f"Не удалось добавить документ {ids[j]}")
                    continue

        # Освобождаем память
        del batch_ids, batch_docs, batch_meta

    progress_bar.close()

    logger.info(f"Успешно добавлено документов: {inserted_count}/{len(ids)}")
    logger.info(f"Векторная база создана по пути: {CHROMA_PATH}")
    logger.info("Процесс завершен успешно!")


if __name__ == "__main__":
    create_chroma_db()