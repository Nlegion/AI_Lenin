import json
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings
from tqdm import tqdm
from chromadb.errors import NotFoundError
import logging
import torch

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Конфигурация путей
DATA_DIR = "data/processed"
DB_DIR = "database/vector_db"
CHUNK_FILE = "lenin_chunks.jsonl"

# Полные пути
chunk_file_path = os.path.join(DATA_DIR, CHUNK_FILE)
db_path = os.path.join(DB_DIR)

# Создаем директории, если их нет
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(db_path, exist_ok=True)

# Инициализация функции эмбеддингов
logger.info("Инициализация модели эмбеддингов...")
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() and os.name != 'nt' else "cpu"
)

# Проверка размерности эмбеддингов
embedding_dim = embedding_function._model.get_sentence_embedding_dimension()
logger.info(f"Размерность эмбеддингов: {embedding_dim}")

# Настройки ChromaDB
client_settings = Settings(
    persist_directory=db_path,
    anonymized_telemetry=False
)

# Создаем клиент ChromaDB
client = chromadb.PersistentClient(path=db_path, settings=client_settings)

# Удаляем существующую коллекцию (если есть)
try:
    client.delete_collection("lenin_works")
    logger.info("Старая коллекция удалена")
except NotFoundError:
    logger.info("Коллекция не найдена, создаем новую")

# Создаем новую коллекцию с правильной размерностью
collection = client.get_or_create_collection(
    name="lenin_works",
    embedding_function=embedding_function,
    metadata={"embedding_dimension": str(embedding_dim)}
)
logger.info(f"Коллекция 'lenin_works' создана с размерностью {embedding_dim}")

# Чтение и обработка чанков
logger.info(f"Загрузка чанков из: {chunk_file_path}")

if not os.path.exists(chunk_file_path):
    logger.error(f"Файл с чанками не найден: {chunk_file_path}")
    exit(1)

# Оптимизация для больших файлов: читаем построчно без загрузки всего файла в память
total_chunks = 0
with open(chunk_file_path, "r", encoding="utf-8") as f:
    for _ in f:
        total_chunks += 1

logger.info(f"Найдено чанков: {total_chunks}")

# Параметры батчей
batch_size = 500  # Увеличиваем размер батча для производительности
documents_batch = []
metadatas_batch = []
ids_batch = []
processed_chunks = 0

# Счетчик для прогресс-бара
progress_bar = tqdm(total=total_chunks, desc="Обработка чанков")

with open(chunk_file_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            text = data.get("text", "")
            metadata = data.get("metadata", {})

            # Генерация ID на основе метаданных
            doc_id = metadata.get("doc_id", f"chunk_{processed_chunks}")

            documents_batch.append(text)
            metadatas_batch.append(metadata)
            ids_batch.append(doc_id)

            # Добавляем батч при достижении размера
            if len(ids_batch) >= batch_size:
                collection.add(
                    documents=documents_batch,
                    metadatas=metadatas_batch,
                    ids=ids_batch
                )
                # Сбрасываем батчи
                documents_batch = []
                metadatas_batch = []
                ids_batch = []

            processed_chunks += 1
            progress_bar.update(1)

        except json.JSONDecodeError:
            logger.error(f"Ошибка JSON в строке {processed_chunks + 1}")
        except Exception as e:
            logger.error(f"Ошибка обработки чанка {processed_chunks + 1}: {str(e)}")

# Добавляем последний батч
if ids_batch:
    collection.add(
        documents=documents_batch,
        metadatas=metadatas_batch,
        ids=ids_batch
    )
    logger.info(f"Добавлен последний батч из {len(ids_batch)} чанков")

progress_bar.close()

logger.info("\n" + "=" * 50)
logger.info(f"Векторная БД успешно создана в: {db_path}")
logger.info(f"Обработано чанков: {processed_chunks}")
logger.info(f"Общее количество в коллекции: {collection.count()}")
logger.info("=" * 50)