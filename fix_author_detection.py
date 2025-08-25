import os
import re
from pathlib import Path


def fix_author_detection():
    """Исправляет определение автора в очищенных файлах"""
    cleaned_dir = Path(r"P:\AI_Lenin\data\books\ultimate_cleaned_ontology")

    if not cleaned_dir.exists():
        print("Директория с очищенными файлами не существует!")
        return

    # Создаем словарь для переименования файлов
    author_mapping = {}

    # Проходим по всем файлам и определяем автора по структуре папок
    for root, dirs, files in os.walk(cleaned_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = Path(root) / file
                relative_path = file_path.relative_to(cleaned_dir)

                # Определяем автора по структуре папок
                parts = relative_path.parts
                if len(parts) > 0:
                    author = parts[0]  # Первая папка - это автор

                    # Специальные случаи
                    if author == "pss":
                        author = "Ленин"
                    elif author == "single":
                        # Для файлов в папке single, автор определяется по родительской папке
                        if len(parts) > 1:
                            author = parts[1]

                    # Добавляем в словарь
                    if author not in author_mapping:
                        author_mapping[author] = []
                    author_mapping[author].append(file_path)

    # Выводим статистику
    print("=" * 60)
    print("СТАТИСТИКА АВТОРОВ В ОЧИЩЕННЫХ ФАЙЛАХ")
    print("=" * 60)

    for author, files in author_mapping.items():
        print(f"{author}: {len(files)} файлов")

    return author_mapping


def add_author_metadata():
    """Добавляет метаданные об авторе в начало каждого файла"""
    cleaned_dir = Path(r"P:\AI_Lenin\data\books\ultimate_cleaned_ontology")

    if not cleaned_dir.exists():
        print("Директория с очищенными файлами не существует!")
        return

    # Проходим по всем файлам и добавляем метаданные об авторе
    for root, dirs, files in os.walk(cleaned_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = Path(root) / file
                relative_path = file_path.relative_to(cleaned_dir)

                # Определяем автора по структуре папок
                parts = relative_path.parts
                if len(parts) > 0:
                    author = parts[0]  # Первая папка - это автор

                    # Специальные случаи
                    if author == "pss":
                        author = "Ленин"
                    elif author == "single":
                        # Для файлов в папке single, автор определяется по родительской папке
                        if len(parts) > 1:
                            author = parts[1]

                    # Читаем содержимое файла
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # Добавляем метаданные об авторе в начало файла
                    new_content = f"АВТОР: {author}\nРАБОТА: {file_path.stem}\n\n{content}"

                    # Перезаписываем файл
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

                    print(f"Добавлены метаданные для: {file_path}")


if __name__ == "__main__":
    # Сначала посмотрим статистику
    author_mapping = fix_author_detection()

    # Затем добавим метаданные об авторе в файлы
    add_author_metadata()

    print("Метаданные об авторе успешно добавлены во все файлы!")