import os

# Конфигурация
root_dir = os.getcwd()  # Стартовая директория (текущий проект)
output_file = "project_structure.txt"  # Имя выходного файла
exclude_dirs = {'.venv', '.git', '__pycache__', '.idea', 'node_modules'}  # Исключаемые папки
exclude_hidden = True  # Исключать скрытые файлы/папки (начинающиеся с '.')

def process_directory(root_dir, output_file, exclude_dirs, exclude_hidden):
    with open(output_file, 'w', encoding='utf-8') as f:
        for current_dir, dirs, files in os.walk(root_dir, topdown=True):
            # Фильтрация исключаемых директорий
            dirs[:] = [
                d for d in dirs
                if not (d in exclude_dirs or (exclude_hidden and d.startswith('.')))
            ]

            rel_path = os.path.relpath(current_dir, root_dir)

            # Пропуск исключаемых директорий
            if any(ex_dir in rel_path.split(os.sep) for ex_dir in exclude_dirs):
                continue

            # Корректировка пути для корневой директории
            if rel_path == ".":
                rel_path = ""

            # Запись директории (если не скрытая)
            if rel_path and not (exclude_hidden and rel_path.startswith('.')):
                f.write(f"{rel_path}/\n")

            # Запись файлов в директории
            for file in files:
                if exclude_hidden and file.startswith('.'):
                    continue
                file_path = os.path.join(rel_path, file)
                if file_path.startswith(f'.{os.sep}'):
                    file_path = file_path[2:]
                f.write(f"{file_path}\n")

# Запуск функции
process_directory(root_dir, output_file, exclude_dirs, exclude_hidden)
print(f"✓ Структура проекта сохранена в: {output_file}")
print(f"✓ Исключенные папки: {', '.join(exclude_dirs)}")
if exclude_hidden:
    print("✓ Скрытые файлы/папки исключены")
