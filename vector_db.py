import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

def initialize_vector_db():
    # Конфигурация
    CHUNKS_FILE = "data/processed/lenin_chunks.jsonl"
    EMBED_MODEL = "intfloat/multilingual-e5-large"

    # Проверка наличия файла
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Файл чанков {CHUNKS_FILE} не найден. Сначала выполните предобработку данных.")

    # Инициализация модели
    embed_model = SentenceTransformer(EMBED_MODEL, device="cuda")

    # Инициализация ChromaDB
    client = chromadb.PersistentClient(path="vector_db")
    collection = client.get_or_create_collection(
        name="lenin_works",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        ),
        metadata={"hnsw:space": "cosine"}  # Оптимизация для семантического поиска
    )

    # Загрузка чанков
    documents = []
    metadatas = []
    ids = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Загрузка чанков"):
            data = json.loads(line)
            documents.append(data["text"])
            metadatas.append(data["metadata"])
            ids.append(data["metadata"]["doc_id"])

    # Пакетная загрузка в ChromaDB
    batch_size = 100
    for i in tqdm(range(0, len(documents), batch_size), desc="Индексация"):
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        embeddings = embed_model.encode(batch_docs).tolist()

        collection.add(
            embeddings=embeddings,
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )

    print(f"Векторная БД инициализирована. Документов: {collection.count()}")


if __name__ == "__main__":
    initialize_vector_db()