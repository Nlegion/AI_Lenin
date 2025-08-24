import json
import re
import random
from tqdm import tqdm
import logging
import os
from datetime import datetime
from llama_cpp import Llama

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Очистка текста от лишних символов и форматирования"""
    if not text:
        return ""
    # Удаление HTML тегов
    text = re.sub(r'<[^>]+>', '', text)
    # Удаление спецсимволов (кроме пунктуации и букв)
    text = re.sub(r'[^\w\s.,!?;:а-яА-ЯёЁ-]', '', text)
    # Замена множественных пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_content(news: dict) -> tuple:
    """Извлекает и очищает заголовок и содержание из разных форматов данных"""
    # Вариант 1: прямой доступ к title и content
    title = clean_text(news.get('title', ''))
    content = clean_text(news.get('content', ''))

    # Вариант 2: обработка формата с полем 'input'
    if not content and 'input' in news:
        input_text = news['input']
        match = re.match(r'Заголовок: (.+?)\nТекст: (.+)', input_text)
        if match:
            title = clean_text(match.group(1))
            content = clean_text(match.group(2))
        else:
            content = clean_text(input_text)

    # Вариант 3: обработка других форматов
    if not content:
        for key in ['text', 'article', 'body']:
            if key in news:
                content = clean_text(news[key])
                break

    return title, content


def generate_system_prompt() -> str:
    """Генерация системного промпта в стиле Ленина"""
    class_conflicts = [
        "эксплуатация пролетариата",
        "борьба за средства производства",
        "противостояние буржуазии и рабочего класса",
        "империалистическая агрессия",
        "революционная ситуация",
        "диктатура пролетариата"
    ]

    revolutionary_phrases = [
        "Пролетарии всех стран, соединяйтесь!",
        "Вся власть Советам!",
        "Мир хижинам, война дворцам!",
        "Социалистическая революция неизбежна!",
        "Капитализм несет угнетение масс!"
    ]

    return (
        f"Ты — Владимир Ильич Ленин. Дай марксистский анализ классовой борьбы "
        f"в 2 предложениях на основе новости. Анализ должен:\n"
        f"- Раскрывать классовые противоречия ({random.choice(class_conflicts)})\n"
        f"- Быть в революционном стиле ({random.choice(revolutionary_phrases)})\n"
        f"- Не упоминать конкретные страны и национальности"
    )


def create_lora_dataset(
        input_file: str,
        output_file: str,
        model_path: str,
        num_samples: int = 500
):
    """
    Создает датасет для обучения LoRA адаптера с генерацией ответов модели

    :param input_file: Путь к файлу с новостями (JSONL формат)
    :param output_file: Путь для сохранения датасета LoRA
    :param model_path: Путь к GGUF модели
    :param num_samples: Количество образцов для датасета
    """
    logger.info(f"Создание датасета LoRA из {input_file} с использованием модели {model_path}")

    # Проверка существования файлов
    if not os.path.exists(input_file):
        logger.error(f"Файл не найден: {input_file}")
        return
    if not os.path.exists(model_path):
        logger.error(f"Модель не найдена: {model_path}")
        return

    # Создание директории для выходного файла
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Загрузка модели
    logger.info("Загрузка модели...")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=50,
        n_threads=min(8, os.cpu_count() or 4),
        verbose=False
    )
    logger.info("Модель успешно загружена")

    # Чтение и обработка новостей
    news_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                news_items.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Ошибка декодирования JSON: {str(e)}")

    if not news_items:
        logger.error("Нет данных для создания датасета")
        return

    logger.info(f"Загружено {len(news_items)} новостей")

    # Случайная выборка
    selected_news = random.sample(news_items, min(num_samples, len(news_items)))
    logger.info(f"Обработка {len(selected_news)} образцов")

    # Создание датасета
    dataset = []
    skipped_count = 0
    for news in tqdm(selected_news, desc="Генерация датасета"):
        try:
            # Извлечение и очистка данных
            title, content = extract_content(news)

            # Проверка наличия контента
            if not content:
                skipped_count += 1
                logger.debug(f"Пропуск новости без контента: {news.get('title', '')}")
                continue

            # Формирование контекста пользователя
            user_content = f"Заголовок: {title}\nТекст: {content[:300]}" if title else f"Текст: {content[:300]}"

            # Генерация системного промпта
            system_prompt = generate_system_prompt()

            # Формирование полного промпта
            dialog_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            # Генерация ответа модели
            response = llm(
                prompt=dialog_prompt,
                max_tokens=150,
                temperature=0.7,
                stop=["<|im_end|>", "\n\n"],
                echo=False
            )

            # Извлечение текста ответа
            analysis = response['choices'][0]['text'].strip()
            analysis = re.sub(r'^Анализ:\s*', '', analysis)

            # Формирование примера для LoRA
            example = {
                "conversations": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": analysis}
                ]
            }

            dataset.append(example)
        except Exception as e:
            logger.error(f"Ошибка обработки новости: {str(e)}")
            skipped_count += 1

    # Сохранение датасета
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"Датасет LoRA сохранен: {output_file}")
    logger.info(f"Успешно обработано: {len(dataset)} примеров")
    logger.info(f"Пропущено: {skipped_count} примеров")
    logger.info(f"Причины пропуска: отсутствие контента или ошибки обработки")


if __name__ == "__main__":
    # Включение детального логирования при необходимости
    logger.setLevel(logging.DEBUG)

    # Конфигурация
    INPUT_DATASET = "data/processed/lenin_dataset.jsonl"
    OUTPUT_DATASET = "data/finetune/lenin_lora_final.jsonl"
    MODEL_PATH = "../models/saiga/legacy/model-q4_K.gguf"

    # Создание датасета
    create_lora_dataset(
        input_file=INPUT_DATASET,
        output_file=OUTPUT_DATASET,
        model_path=MODEL_PATH,
        num_samples=1000
    )