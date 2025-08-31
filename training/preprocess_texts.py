import os
import json
import re
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Очистка текста: удаление лишних пробелов, переносов, непечатаемых символов,
    номеров страниц и специальных пометок, характерных для собраний сочинений.
    """
    # Удаление номеров страниц в формате [123]
    text = re.sub(r'\[\d+\]', '', text)
    # Удаление сносок вида {123}
    text = re.sub(r'\{\d+\}', '', text)
    # Удаление лишних переносов строк
    text = re.sub(r'\n+', '\n', text)
    # Удаление последовательностей из 3+ повторяющихся символов (кроме букв)
    text = re.sub(r'([^a-zA-Zа-яА-Я])\1{2,}', r'\1', text)
    # Замена множественных пробелов
    text = re.sub(r'\s+', ' ', text)
    # Удаление управляющих символов
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    return text.strip()


def process_files(
        input_dir: str,
        output_file: str,
        tokenizer_name: str = "psuplj/Meta-Llama-3-8B-Q4_K_M-GGUF",
        max_length: int = 2048,
        min_length: int = 256,
        skip_special_tokens: bool = True
):
    """
    Обработка файлов с текстами и создание датасета в формате JSONL.

    Параметры:
    input_dir: Папка с исходными TXT-файлами
    output_file: Выходной JSONL-файл
    tokenizer_name: Имя или путь к токенизатору
    max_length: Максимальная длина фрагмента в токенах
    min_length: Минимальная длина фрагмента для сохранения
    skip_special_tokens: Пропускать спецтокены при декодировании
    """
    # Создаем директории, если их нет
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Инициализация токенизатора
    try:
        logger.info(f"Загрузка токенизатора: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logger.error(f"Ошибка загрузки токенизатора: {e}")
        logger.info("Используем резервный токенизатор для русского языка")
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base")

    # Для Llama 3 используем специальные токены
    if "llama-3" in tokenizer_name.lower():
        tokenizer.add_special_tokens({
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": "<|end_of_text|>"
        })

    chunk_size = max_length - 2  # Резерв для [BOS] и [EOS] токенов
    total_chunks = 0
    total_files = sum(1 for f in os.listdir(input_dir) if f.endswith('.txt'))

    with open(output_file, 'w', encoding='utf-8') as f_out:
        file_list = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

        for filename in tqdm(file_list, desc="Обработка файлов"):
            filepath = os.path.join(input_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f_in:
                    text = clean_text(f_in.read())

                    # Пропускаем пустые файлы
                    if not text.strip():
                        logger.warning(f"Пустой файл: {filename}")
                        continue

                    # Токенизация всего текста
                    tokens = tokenizer.encode(text, add_special_tokens=False)

                    # Разбивка на блоки по chunk_size токенов
                    for i in range(0, len(tokens), chunk_size):
                        chunk = tokens[i:i + chunk_size]
                        if len(chunk) < min_length:
                            continue

                        # Декодирование фрагмента
                        text_chunk = tokenizer.decode(chunk, skip_special_tokens=skip_special_tokens)

                        # Запись в формате JSONL
                        json_line = json.dumps({"text": text_chunk}, ensure_ascii=False)
                        f_out.write(json_line + '\n')
                        total_chunks += 1

                logger.debug(f"Обработан: {filename} | Фрагментов: {len(tokens) // chunk_size + 1}")
            except Exception as e:
                logger.error(f"Ошибка обработки файла {filename}: {str(e)}")

    logger.info(f"\nОбработка завершена!")
    logger.info(f"Всего обработано файлов: {len(file_list)}")
    logger.info(f"Сгенерировано фрагментов: {total_chunks}")
    logger.info(f"Датасет сохранен в: {output_file}")


if __name__ == "__main__":
    # Конфигурация путей
    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_DIR = BASE_DIR / "data" / "raw"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "lenin_dataset.jsonl"

    # Проверка существования исходных данных
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Директория с исходными данными не найдена: {INPUT_DIR}")

    logger.info("Начало подготовки датасета...")
    logger.info(f"Исходная директория: {INPUT_DIR}")
    logger.info(f"Выходной файл: {OUTPUT_FILE}")

    # Используем публичный токенизатор, совместимый с Llama 3
    process_files(
        input_dir=str(INPUT_DIR),
        output_file=str(OUTPUT_FILE),
        tokenizer_name="psuplj/Meta-Llama-3-8B-Q4_K_M-GGUF",
        max_length=2048,
        min_length=256,
        skip_special_tokens=True
    )