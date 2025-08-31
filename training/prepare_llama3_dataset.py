import os
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer


def clean_text(text: str) -> str:
    """
    Очистка текста: удаление лишних пробелов, переносов, непечатаемых символов.
    Сохранение структуры предложений и пунктуации.
    """
    text = re.sub(r'\s+', ' ', text)  # Замена множественных пробелов
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Удаление управляющих символов
    text = re.sub(r'\n+', '\n', text)  # Удаление лишних переносов строк
    return text.strip()


def process_files(
        input_dir: str,
        output_file: str,
        tokenizer_name: str = "meta-llama/Meta-Llama-3-8B",
        max_length: int = 2048,
        min_length: int = 100
):
    """
    Обработка файлов с текстами и создание датасета в формате JSONL.

    Параметры:
    input_dir: Папка с исходными TXT-файлами
    output_file: Выходной JSONL-файл
    tokenizer_name: Идентификатор токенизатора
    max_length: Максимальная длина фрагмента в токенах (с учетом спецтокенов)
    min_length: Минимальная длина фрагмента для сохранения
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    chunk_size = max_length - 2  # Резерв для [BOS] и [EOS] токенов

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for filename in tqdm(os.listdir(input_dir)):
            if filename.endswith('.txt'):
                filepath = os.path.join(input_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f_in:
                    text = clean_text(f_in.read())
                    tokens = tokenizer.encode(text, add_special_tokens=False)

                    # Разбивка на блоки по chunk_size токенов
                    for i in range(0, len(tokens), chunk_size):
                        chunk = tokens[i:i + chunk_size]
                        if len(chunk) < min_length:
                            continue

                        # Декодирование с пропуском спецтокенов
                        text_chunk = tokenizer.decode(chunk, skip_special_tokens=True)

                        # Запись в формате JSONL
                        json_line = json.dumps({"text": text_chunk}, ensure_ascii=False)
                        f_out.write(json_line + '\n')


if __name__ == "__main__":
    # Конфигурация путей
    INPUT_DIR = "lenin_toms"  # Папка с исходными текстами
    OUTPUT_FILE = "lenin_dataset.jsonl"  # Выходной файл

    process_files(
        input_dir=INPUT_DIR,
        output_file=OUTPUT_FILE,
        max_length=2048,  # Соответствует стандартному контексту Llama3
        min_length=100  # Минимальная длина фрагмента
    )