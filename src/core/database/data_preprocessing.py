import os
import re
import json
import uuid
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Конфигурация
INPUT_DIR = "../../../data/raw"
OUTPUT_FILE = "../../../data/processed/lenin_chunks.jsonl"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_volume_number(filename: str) -> int:
    match = re.search(r'том[\s_]*(\d+)', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def clean_text(text: str) -> str:
    text = re.sub(r'\[\s*(\d+|стр\.\s*\d+)\s*\]', '', text)
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_file(file_path: str, volume: int, writer):
    loader = TextLoader(file_path, autodetect_encoding=True)
    document = loader.load()[0]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". "]
    )

    chunks = splitter.split_text(document.page_content)

    for i, chunk in enumerate(chunks):
        metadata = {
            "source_title": f"ПСС Ленина, Том {volume}",
            "volume": volume,
            "chunk_index": i,
            "doc_id": f"lenin_vol{volume}_chunk{i}_{uuid.uuid4().hex[:6]}"
        }

        writer.write(json.dumps({
            "text": clean_text(chunk),
            "metadata": metadata
        }, ensure_ascii=False) + "\n")


def preprocess_data():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as writer:
        for filename in tqdm(files, desc="Обработка томов"):
            file_path = os.path.join(INPUT_DIR, filename)
            volume = extract_volume_number(filename)
            if volume:
                process_file(file_path, volume, writer)


if __name__ == "__main__":
    preprocess_data()