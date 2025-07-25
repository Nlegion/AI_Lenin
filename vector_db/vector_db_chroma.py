import chromadb
from data_prep import prepare_dataset
from sentence_transformers import SentenceTransformer

# Инициализация модели для эмбеддингов
embed_model = SentenceTransformer(
    'intfloat/multilingual-e5-large',
    device='cuda'
)


def create_chroma_db():
    # Подготовка данных
    texts, metadata = prepare_dataset()

    # Генерация эмбеддингов
    embeddings = embed_model.encode(texts, show_progress_bar=True)

    # Создание коллекции Chroma
    client = chromadb.PersistentClient(path="lenin_chroma_db")
    collection = client.create_collection(name="lenin_works")

    # Добавление документов
    ids = [f"doc_{i}" for i in range(len(texts))]

    collection.add(
        embeddings=[emb.tolist() for emb in embeddings],
        documents=texts,
        metadatas=metadata,
        ids=ids
    )

    print(f"База Chroma создана с {len(texts)} документами")


if __name__ == "__main__":
    create_chroma_db()