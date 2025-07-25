from qdrant_client import QdrantClient, models
from data_prep import prepare_dataset
from sentence_transformers import SentenceTransformer


def create_qdrant_db():
    # Подготовка данных
    texts, metadata = prepare_dataset()

    # Инициализация клиента
    client = QdrantClient(":memory:")  # Для постоянного хранилища: QdrantClient(path="./qdrant_db")

    # Создание коллекции
    client.create_collection(
        collection_name="lenin_collection",
        vectors_config=models.VectorParams(
            size=1024,  # Размерность для multilingual-e5-large
            distance=models.Distance.COSINE
        )
    )

    # Генерация и загрузка эмбеддингов
    embed_model = SentenceTransformer('intfloat/multilingual-e5-large', device='cuda')
    embeddings = embed_model.encode(texts, show_progress_bar=True)

    client.upload_points(
        collection_name="lenin_collection",
        points=[
            models.PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={
                    "text": text,
                    "metadata": meta
                }
            )
            for idx, (text, emb, meta) in enumerate(zip(texts, embeddings, metadata))
        ]
    )

    print(f"База Qdrant создана с {len(texts)} документами")


if __name__ == "__main__":
    create_qdrant_db()