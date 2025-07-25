import os
import re
import json
import uuid
from typing import List, Dict, Tuple
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Конфигурационные параметры
INPUT_DIR = "data/raw"
OUTPUT_FILE = "data/processed/lenin_chunks.jsonl"
BASE_URL = "https://lenin.org/tom{volume}/page{page}"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_SENTENCE_LENGTH = 50

# Ключевые термины для тематической разметки
THEMATIC_TAGS = {
    "империализм": ["империализм", "монополия", "капитал"],
    "революция": ["революция", "восстание", "пролетариат"],
    "партия": ["партия", "большевик", "организация"],
    "государство": ["государство", "диктатура", "аппарат"],
    "философия": ["диалектика", "материализм", "идеализм"]
}


def extract_volume_number(filename: str) -> int:
    """Извлекает номер тома из имени файла"""
    match = re.search(r'том[\s_]*(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def clean_text(text: str) -> str:
    """Очистка текста от сносок и артефактов с сохранением структуры"""
    # Удаление редакторских сносок [1], [стр. 34] но сохранение [1] в конце предложений
    text = re.sub(r'\[\s*(\d+|стр\.\s*\d+|\w+\.\d+)\s*\]', '', text)

    # Удаление технических пометок, но сохранение важных примечаний
    text = re.sub(r'\b(рис|илл?|табл)\.?\s*\d*', '', text, flags=re.IGNORECASE)

    # Удаление переносов слов
    text = re.sub(r'-\s*\n\s*', '', text)

    # Нормализация пробелов и переносов строк
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(\n\s*){2,}', '\n\n', text)

    # Восстановление поврежденных слов
    text = re.sub(r'(\w)-\s*(\w)', r'\1\2', text)

    return text.strip()


def extract_pages(text: str) -> List[Tuple[int, str]]:
    """Извлекает страницы из текста с улучшенной обработкой маркеров"""
    pages = []
    current_page = 1
    page_start = 0
    last_valid_page = 1

    # Улучшенное регулярное выражение для поиска маркеров страниц
    page_markers = re.compile(
        r'(\[\s*стр\.?\s*(\d+)\s*\]|'  # [стр. N]
        r'\bстр\s*[.:]?\s*(\d+)\b|'  # стр N
        r'\bс\.\s*(\d+)\b|'  # с. N (сокращение)
        r'^\s*[\[({]?\s*(\d+)\s*[\]})]?\s*$)'  # Изолированные цифры
    )

    for match in page_markers.finditer(text):
        marker_start, marker_end = match.span()
        page_num = None

        # Извлечение номера страницы из соответствующей группы
        for i in range(2, 6):  # Группы 2-5 содержат номера страниц
            if match.group(i) and match.group(i).isdigit():
                page_num = int(match.group(i))
                break

        if page_num is None:
            continue

        # Текст между маркерами - предыдущая страница
        page_text = text[page_start:marker_start].strip()

        # Добавляем страницу только если есть текст и номер страницы валиден
        if page_text and len(page_text) > MIN_SENTENCE_LENGTH:
            # Проверяем последовательность нумерации
            if page_num > current_page + 50:  # Слишком большой скачок
                page_num = current_page + 1
            elif page_num < current_page - 5:  # Невероятный возврат назад
                page_num = last_valid_page

            pages.append((current_page, page_text))
            last_valid_page = current_page
            current_page = page_num
        page_start = marker_end

    # Обработка остатка текста после последнего маркера
    final_text = text[page_start:].strip()
    if final_text and len(final_text) > MIN_SENTENCE_LENGTH:
        pages.append((current_page, final_text))

    # Коррекция нумерации при отсутствии маркеров
    if not pages:
        pages.append((1, text))

    return pages


def assign_thematic_tags(text: str) -> List[str]:
    """Назначает тематические теги на основе ключевых слов"""
    tags = []
    for tag, keywords in THEMATIC_TAGS.items():
        if any(keyword in text.lower() for keyword in keywords):
            tags.append(tag)
    return tags


def process_file(file_path: str, volume: int, writer):
    """Обрабатывает один файл тома"""
    loader = TextLoader(file_path, autodetect_encoding=True)
    document = loader.load()[0]

    pages = extract_pages(document.page_content)
    cleaned_pages = []

    for page_num, text in pages:
        cleaned = clean_text(text)
        if cleaned:
            cleaned_pages.append((page_num, cleaned))

    text_with_boundaries = []
    for page_num, text in cleaned_pages:
        text_with_boundaries.append(f"[PAGE:{page_num}]\n{text}")
    full_text = "\n\n".join(text_with_boundaries)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", r"(?<=\. )", r"(?<=\? )", r"(?<=\! )", " ", ""],
        keep_separator=True,
        is_separator_regex=True
    )

    chunks = splitter.split_text(full_text)
    doc_id_prefix = f"lenin_vol{volume}_"

    for chunk in chunks:
        page_matches = list(re.finditer(r'\[PAGE:(\d+)\]', chunk))
        if not page_matches:
            continue

        page_numbers = [int(m.group(1)) for m in page_matches]
        start_page = min(page_numbers)
        end_page = max(page_numbers)

        clean_chunk = re.sub(r'\[PAGE:\d+\]\n?', '', chunk).strip()
        if not clean_chunk:
            continue

        # Назначение тематических тегов
        thematic_tags = assign_thematic_tags(clean_chunk)

        metadata = {
            "source_title": f"ПСС Ленина, Том {volume}",
            "volume": volume,
            "page_start": start_page,
            "page_end": end_page,
            "doc_id": f"{doc_id_prefix}start{start_page}_end{end_page}_{str(uuid.uuid4())[:6]}",
            "url": BASE_URL.format(volume=volume, page=start_page),
            "thematic_tags": ", ".join(thematic_tags)  # Строка вместо списка
        }

        writer.write(json.dumps({
            "text": clean_chunk,
            "metadata": metadata
        }, ensure_ascii=False) + "\n")


def preprocess_data():
    """Основная функция обработки данных"""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as writer:
        for filename in tqdm(files, desc="Обработка томов"):
            file_path = os.path.join(INPUT_DIR, filename)
            volume = extract_volume_number(filename)

            if volume is None:
                print(f"Не удалось определить том для файла: {filename}")
                continue

            try:
                process_file(file_path, volume, writer)
            except Exception as e:
                print(f"Ошибка обработки файла {filename}: {str(e)}")

    # Проверка уникальности идентификаторов
    validate_doc_ids()


def validate_doc_ids():
    """Проверка уникальности doc_id"""
    doc_ids = set()
    total_chunks = 0

    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            total_chunks += 1
            data = json.loads(line)
            doc_id = data['metadata']['doc_id']
            if doc_id in doc_ids:
                raise ValueError(f"Найдены дубликаты doc_id: {doc_id}")
            doc_ids.add(doc_id)

    print(f"Валидация пройдена: все {total_chunks} идентификаторов уникальны")


if __name__ == "__main__":
    preprocess_data()
    print(f"Обработка завершена. Результаты сохранены в {OUTPUT_FILE}")