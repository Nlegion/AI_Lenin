import os
from pathlib import Path
import re
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_ultimate_cleaning():
    """Анализирует качество ультимативной очистки"""
    original_dir = Path(r"/data/books/intellectual")
    ultimate_dir = Path(r"/data/books/ultimate_cleaned_ontology")

    if not ultimate_dir.exists():
        print("Директория с ультимативной очисткой не существует!")
        return

    # Находим все файлы для анализа
    original_files = list(original_dir.rglob("*.txt"))

    print("=" * 80)
    print("АНАЛИЗ КАЧЕСТВА УЛЬТИМАТИВНОЙ ОЧИСТКИ")
    print("=" * 80)

    total_original = 0
    total_ultimate = 0

    for original_file in original_files:
        relative_path = original_file.relative_to(original_dir)
        ultimate_file = ultimate_dir / relative_path

        if not ultimate_file.exists():
            print(f"Ультимативный файл не найден: {relative_path}")
            continue

        # Читаем файлы
        with open(original_file, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()

        with open(ultimate_file, 'r', encoding='utf-8') as f:
            ultimate_content = f.read()

        # Анализ
        original_size = len(original_content)
        ultimate_size = len(ultimate_content)

        total_original += original_size
        total_ultimate += ultimate_size

        if original_size > 0:
            reduction = (1 - ultimate_size / original_size) * 100
        else:
            reduction = 0

        print(f"{relative_path}: {original_size} → {ultimate_size} символов ({reduction:.1f}% reduction)")

        # Проверяем на наличие технических артефактов
        technical_patterns = [
            r"Тираж", r"Цена", r"Редактор", r"Заведующий редакцией",
            r"Подписано к печати", r"Сдано в набор", r"ISBN", r"©",
            r"том\s*\d+", r"Том\s*\d+", r"стр\.\s*\d+", r"МОСКВА\s*\d{4}",
            r"ПЕЧАТАЕТСЯ ПО ПОСТАНОВЛЕНИЮ", r"ИНСТИТУТ МАРКСИЗМА-ЛЕНИНИЗМА"
        ]

        artifacts_found = []
        for pattern in technical_patterns:
            if re.search(pattern, ultimate_content, re.IGNORECASE):
                artifacts_found.append(pattern)

        if artifacts_found:
            print(f"  Найдены артефакты: {', '.join(artifacts_found[:2])}" +
                  ("..." if len(artifacts_found) > 2 else ""))

    # Общая статистика
    print("\n" + "=" * 80)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 80)
    print(f"Общий объем оригинальных файлов: {total_original} символов")
    print(f"Общий объем ультимативных файлов: {total_ultimate} символов")

    if total_original > 0:
        total_reduction = (1 - total_ultimate / total_original) * 100
        print(f"Общее сокращение: {total_reduction:.1f}%")

    # Проверяем несколько случайных файлов для качественного анализа
    print("\n" + "=" * 80)
    print("КАЧЕСТВЕННЫЙ АНАЛИЗ (случайные файлы)")
    print("=" * 80)

    import random
    sample_files = random.sample(original_files, min(5, len(original_files)))

    for original_file in sample_files:
        relative_path = original_file.relative_to(original_dir)
        ultimate_file = ultimate_dir / relative_path

        if not ultimate_file.exists():
            continue

        with open(ultimate_file, 'r', encoding='utf-8') as f:
            ultimate_content = f.read()

        print(f"\n{relative_path}:")
        print("Начало очищенного файла:")
        if len(ultimate_content) > 500:
            print(ultimate_content[:500] + "...")
        else:
            print(ultimate_content)
        print("-" * 60)


if __name__ == "__main__":
    analyze_ultimate_cleaning()