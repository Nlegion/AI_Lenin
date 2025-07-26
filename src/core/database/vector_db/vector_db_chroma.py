import chromadb
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = "../../../data/processed/lenin_chunks.jsonl"
CHROMA_PATH = "../../../database/vector_db"


def create_chroma_db():
    embed_model = SentenceTransformer('intfloat/multilingual-e5-large', device='cuda')
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="lenin_works",
        embedding_function=embed_model.encode
    )

    documents = []
    metadatas = []
    ids = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Загрузка данных"):
            data = json.loads(line)
            documents.append(data['text'])
            metadatas.append(data['metadata'])
            ids.append(data['metadata']['doc_id'])

    # Пакетная вставка
    for i in tqdm(range(0, len(ids), 100), desc="Добавление в ChromaDB"):
        batch_ids = ids[i:i + 100]
        batch_docs = documents[i:i + 100]
        batch_meta = metadatas[i:i + 100]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )

    print(f"База Chroma создана с {len(ids)} документами")


if __name__ == "__main__":
    create_chroma_db()