import json
import re
import logging
import numpy as np
import pandas as pd
import faiss
import os
from sentence_transformers import SentenceTransformer
from collections import Counter
from tqdm import tqdm
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    filename='stage4_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Добавляем вывод в консоль

# Конфигурация
RAW_DATA_DIR = "data/raw"
TEMPLATE_PATH = "data/stage3/lenin_worldview/lenin_worldview_template.json"
OUTPUT_PATH = "data/stage4/lenin_worldview_dynamic_model.json"
VOLUME_COUNT = 55
BATCH_SIZE = 8
TOP_CONCEPTS_PER_SECTION = 50
TOP_CONCEPTS_PER_VOLUME = 10


def load_template():
    """Загрузка шаблона мировоззрения из этапа 3"""
    try:
        with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            template = json.load(f)
        logger.info(f"Шаблон мировоззрения загружен, концептов: {len(template['unique_patterns'])}")
        return template
    except Exception as e:
        logger.error(f"Ошибка загрузки шаблона: {str(e)}")
        raise


def initialize_model():
    """Инициализация модели для векторного кодирования"""
    try:
        model = SentenceTransformer(
            'paraphrase-multilingual-mpnet-base-v2',
            device='cuda',
            truncate_dim=512
        )
        model.half()  # FP16 для экономии памяти
        logger.info("Модель инициализирована (FP16)")
        return model
    except Exception as e:
        logger.error(f"Ошибка инициализации модели: {str(e)}")
        raise


def build_faiss_index(template, model):
    """Построение индекса FAISS для концептов с кодированием текста"""
    try:
        # Извлекаем концепты из шаблона
        concepts = [item['concept'] for item in template['unique_patterns']]

        # Кодируем концепты в эмбеддинги
        concept_embeddings = model.encode(
            concepts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Создаем индекс FAISS
        index = faiss.IndexFlatIP(concept_embeddings.shape[1])
        index.add(concept_embeddings.astype(np.float32))

        logger.info(f"Индекс FAISS построен, концептов: {index.ntotal}")
        return index, concepts  # Возвращаем также список концептов для сопоставления

    except Exception as e:
        logger.error(f"Ошибка построения индекса: {str(e)}")
        raise


def get_year_from_volume(volume_id):
    # Заменить линейную интерполяцию на реальные даты
    real_years = {
        1: 1893, 5: 1899, 10: 1903, 15: 1907,
        20: 1911, 25: 1915, 30: 1917, 35: 1919,
        40: 1920, 45: 1922, 50: 1923, 55: 1924
    }
    return real_years.get(volume_id, 1893 + (volume_id-1)*0.56)


def is_excluded_section(section_text):
    """Проверка, является ли раздел неинформативным"""
    excluded_keywords = [
        "библиография", "указатель", "примечания",
        "комментарии", "список литературы", "оглавление",
        "содержание", "приложения", "иллюстрации"
    ]

    # Первые 3 строки для анализа
    lines = section_text.splitlines()
    first_lines = " ".join(lines[:3]).lower() if lines else ""

    # Причины исключения
    reasons = []

    # Проверка на короткие разделы (увеличено до 2000 символов)
    if len(section_text) < 2000:
        reasons.append("length<2000")

    # Проверка ключевых слов
    for keyword in excluded_keywords:
        if keyword in first_lines:
            reasons.append(f"keyword:{keyword}")

    # Проверка на технические разделы (номера страниц)
    if re.search(r"стр[а-я]*\.?\s*\d+", first_lines):
        reasons.append("page_number")

    # Проверка на подписи/даты в конце документа
    if len(lines) > 3 and re.search(r"\d{1,2}\s*[а-я]+\s*\d{4}", lines[-1]):
        reasons.append("signature/date")

    return reasons if reasons else False


def process_volume(model, index, concept_names, volume_path, volume_id):
    """Обработка одного тома ПСС с улучшенной диагностикой"""
    try:
        # Загрузка текста с проверкой существования файла
        if not os.path.exists(volume_path):
            logger.error(f"Файл тома {volume_id} не найден: {volume_path}")
            return {
                'volume_id': volume_id,
                'error': f"File not found: {volume_path}"
            }

        with open(volume_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Проверка на пустой файл
        if len(text.strip()) < 1000:
            logger.warning(f"Том {volume_id} слишком мал: {len(text)} символов")
            return {
                'volume_id': volume_id,
                'year': get_year_from_volume(volume_id),
                'sections': [],
                'avg_transformation': 0.0,
                'concept_frequency': {}
            }

        # Сохраняем оригинальные переводы строк
        cleaned_text = re.sub(r'\r\n', '\n', text)  # Нормализация переводов строк
        cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)  # Убираем лишние пустые строки

        # Разделение на разделы (по двойному переводу строки)
        sections = re.split(r'\n{3,}|\f|Глава [IVXLCDM]+|Раздел [IVXLCDM]+|Статья \d+|§ \d+', text)
        logger.info(f"Том {volume_id}: найдено разделов: {len(sections)}")

        filtered_sections = []
        exclusion_stats = Counter()

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # Анализ причины исключения
            exclude_reasons = is_excluded_section(section)

            if exclude_reasons:
                for reason in exclude_reasons:
                    exclusion_stats[reason] += 1

                # Логируем первый исключенный раздел для диагностики
                if i == 0:
                    sample = section[:300] + "..." if len(section) > 300 else section
                    logger.debug(f"Исключен раздел 0: Причины: {exclude_reasons}\nОбразец: '{sample}'")
                continue

            filtered_sections.append(section)

        logger.info(
            f"Том {volume_id}: исключено разделов: {sum(exclusion_stats.values())}, причины: {dict(exclusion_stats)}")
        logger.info(f"Том {volume_id}: осталось разделов: {len(filtered_sections)}")

        if not filtered_sections:
            logger.warning(f"Том {volume_id} не содержит значимых разделов")
            return {
                'volume_id': volume_id,
                'year': get_year_from_volume(volume_id),
                'sections': [],
                'avg_transformation': 0.0,
                'concept_frequency': {}
            }

        # Кодирование разделов батчами
        section_embeddings = model.encode(
            filtered_sections,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Поиск релевантных концептов
        distances, indices = index.search(
            section_embeddings.astype(np.float32),
            TOP_CONCEPTS_PER_SECTION
        )

        # Обработка результатов
        volume_results = []
        concept_counter = Counter()

        for i, (D, I) in enumerate(zip(distances, indices)):
            # Анализ топовых концептов
            top_concepts = []
            for idx, score in zip(I, D):
                concept = concept_names[idx]
                top_concepts.append({
                    'concept': concept,
                    'similarity': float(score)
                })
                concept_counter[concept] += 1

            # Расчёт индекса трансформации
            transformation_index = float(np.mean(D))

            volume_results.append({
                'section_id': f"{volume_id}_{i + 1}",
                'top_concepts': top_concepts,
                'transformation_index': transformation_index
            })

        # Расчёт метрик для тома
        avg_transformation = float(np.mean([s['transformation_index'] for s in volume_results]))
        top_volume_concepts = concept_counter.most_common(TOP_CONCEPTS_PER_VOLUME)

        return {
            'volume_id': volume_id,
            'year': get_year_from_volume(volume_id),
            'sections': volume_results,
            'avg_transformation': avg_transformation,
            'concept_frequency': dict(top_volume_concepts)
        }

    except Exception as e:
        logger.exception(f"Критическая ошибка обработки тома {volume_id}")
        return {
            'volume_id': volume_id,
            'error': str(e)
        }


def analyze_evolution(volumes_data):
    """Анализ эволюции мировоззрения по томам"""
    evolution = []
    concept_evolution = {}

    # Сбор данных по томам
    for volume in volumes_data:
        if 'error' in volume or not volume.get('sections'):
            continue

        evolution.append({
            'volume_id': volume['volume_id'],
            'year': volume['year'],
            'avg_transformation': volume['avg_transformation'],
            'top_concepts': list(volume['concept_frequency'].keys())
        })

        # Отслеживание эволюции концептов
        for concept, freq in volume['concept_frequency'].items():
            if concept not in concept_evolution:
                concept_evolution[concept] = []
            concept_evolution[concept].append({
                'volume': volume['volume_id'],
                'year': volume['year'],
                'frequency': freq
            })

    # Выявление ключевых изменений
    turning_points = []
    prev_transformation = None

    for i in range(1, len(evolution)):
        delta = evolution[i]['avg_transformation'] - evolution[i - 1]['avg_transformation']

        # Порог значительного изменения (15%)
        if abs(delta) > 0.06:
            turning_points.append({
                'volume_id': evolution[i]['volume_id'],
                'year': evolution[i]['year'],
                'delta': float(delta),
                'top_concepts': evolution[i]['top_concepts']
            })

    # Анализ долгосрочных тенденций
    long_term_trends = {}
    for concept, data in concept_evolution.items():
        if len(data) > 10:  # Концепты, встречающиеся в 10+ томах
            first = data[0]['frequency']
            last = data[-1]['frequency']
            change = (last - first) / max(1, first)
            long_term_trends[concept] = {
                'first_volume': data[0]['volume'],
                'last_volume': data[-1]['volume'],
                'change_percent': change * 100
            }

    return {
        'timeline': evolution,
        'concept_evolution': concept_evolution,
        'turning_points': turning_points,
        'long_term_trends': long_term_trends
    }

def main():
    """Основной пайплайн обработки"""
    logger.info("===== ЗАПУСК ЭТАПА 4 =====")

    # Проверка директорий
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Инициализация компонентов
    template = load_template()
    model = initialize_model()
    index, concept_names = build_faiss_index(template, model)  # Исправлено здесь

    # Обработка томов
    volumes_data = []
    volume_paths = []
    for i in range(1, VOLUME_COUNT + 1):
        possible_names = [
            f"том_{i}.txt",
            f"том {i}.txt",
            f"tom_{i}.txt",
            f"volume_{i}.txt",
            f"v{i}.txt",
            f"{i}.txt"
        ]
        for name in possible_names:
            path = os.path.join(RAW_DATA_DIR, name)
            if os.path.exists(path):
                volume_paths.append(path)
                break
        else:
            logger.warning(f"Файл тома {i} не найден")
            volume_paths.append(None)

    for volume_id, volume_path in enumerate(tqdm(volume_paths, desc="Обработка томов"), start=1):
        if not os.path.exists(volume_path):
            logger.warning(f"Файл тома {volume_id} не найден: {volume_path}")
            continue

        result = process_volume(model, index, concept_names, volume_path, volume_id)  # Исправлено здесь
        volumes_data.append(result)

        # Логирование прогресса
        if 'error' in result:
            logger.error(f"Том {volume_id} завершен с ошибкой: {result['error']}")
        else:
            logger.info(
                f"Том {volume_id} обработан: {len(result['sections'])} разделов, "
                f"трансформация: {result['avg_transformation']:.3f}"
            )

    # Анализ эволюции
    evolution_analysis = analyze_evolution(volumes_data)

    # Формирование финальной модели
    worldview_model = {
        'metadata': {
            **template['metadata'],
            'processing_date': pd.Timestamp.now().isoformat(),
            'volumes_processed': VOLUME_COUNT,
            'total_sections': sum(len(v['sections']) for v in volumes_data if 'sections' in v),
            'total_concepts': len(concept_names)  # Добавлено для информации
        },
        'concept_network': template['unique_patterns'],
        'volume_analysis': volumes_data,
        'evolution': evolution_analysis
    }

    # Сохранение результатов
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(worldview_model, f, ensure_ascii=False, indent=2)

    logger.info(f"Обработка завершена. Результаты сохранены в {OUTPUT_PATH}")
    logger.info(f"Томов обработано: {len(volumes_data)}")
    logger.info(f"Точек изменения мировоззрения выявлено: {len(evolution_analysis['turning_points'])}")


if __name__ == "__main__":
    # Временно увеличим уровень логирования для диагностики
    logger.setLevel(logging.DEBUG)

    # Проверка существования raw директории
    if not os.path.exists(RAW_DATA_DIR):
        logger.error(f"Директория с исходными данными не найдена: {RAW_DATA_DIR}")
        exit(1)

    # Проверка наличия томов
    volume_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".txt")]
    logger.info(f"Найдено файлов томов: {len(volume_files)}")

    # Запуск основного пайплайна
    main()