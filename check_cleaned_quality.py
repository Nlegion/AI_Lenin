import os
from pathlib import Path
import random


def check_cleaned_files():
    """Проверяет качество очищенных файлов"""
    cleaned_dir = Path(r"P:\AI_Lenin\data\books\ultimate_cleaned_ontology")

    if not cleaned_dir.exists():
        print("Директория с очищенными файлами не существует!")
        return

    # Находим несколько случайных файлов для проверки
    text_files = list(cleaned_dir.rglob("*.txt"))

    if not text_files:
        print("Не найдено очищенных файлов!")
        return

    # Выбираем 5 случайных файлов для проверки
    sample_files = random.sample(text_files, min(5, len(text_files)))

    print("=" * 80)
    print("ПРОВЕРКА КАЧЕСТВА ОЧИЩЕННЫХ ФАЙЛОВ")
    print("=" * 80)

    for file_path in sample_files:
        print(f"\nФайл: {file_path.relative_to(cleaned_dir)}")
        print("-" * 60)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Показываем первые 500 символов
            preview = content[:500]
            if len(content) > 500:
                preview += "..."

            print(preview)
            print(f"Длина файла: {len(content)} символов")
            print(f"Количество строк: {len(content.splitlines())}")

            # Проверяем наличие технической информации
            technical_indicators = [
                "Гатчинская ул., 26",
                "Главполиграфпрома",
                "Комитета по печати",
                "имени А. М. Горького",
                "Совета Министров СССР",
                "Заведующий редакцией",
                "Редактор",
                "Художественный редактор",
                "Технический редактор",
                "Корректор",
                "Сдано в набор",
                "Подписано к печати",
                "Тираж",
                "Цена",
                "©",
                "ISBN"
            ]

            technical_count = 0
            for indicator in technical_indicators:
                if indicator in content:
                    technical_count += 1

            print(f"Найдено технических индикаторов: {technical_count}")

        except Exception as e:
            print(f"Ошибка чтения файла: {e}")


if __name__ == "__main__":
    check_cleaned_files()
