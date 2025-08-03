import os

# Конфигурация
root_dir = os.getcwd()  # Стартовая директория (текущий проект)
output_file = "project_structure.txt"  # Имя выходного файла
exclude_dirs = {'.venv', '.git', '__pycache__', '.idea', 'node_modules'}  # Исключаемые папки
exclude_hidden = True  # Исключать скрытые файлы/папки (начинающиеся с '.')

with open(output_file, 'w', encoding='utf-8') as f:
    # Рекурсивный обход файловой системы
    for current_dir, dirs, files in os.walk(root_dir, topdown=True):
        # Фильтрация исключаемых директорий
        dirs[:] = [
            d for d in dirs
            if not (d in exclude_dirs or (exclude_hidden and d.startswith('.')))
        ]

        rel_path = os.path.relpath(current_dir, root_dir)

        # Пропускаем исключенные директории
        if any(ex_dir in rel_path for ex_dir in exclude_dirs):
            continue

        # Корректируем путь для корневой директории
        if rel_path == ".":
            rel_path = ""

        # Записываем директорию (если не скрытая)
        if rel_path and not (exclude_hidden and rel_path.startswith('.')):
            f.write(f"{rel_path}/\n")

        # Записываем файлы в директории
        for file in files:
            # Пропускаем скрытые файлы
            if exclude_hidden and file.startswith('.'):
                continue

            file_path = os.path.join(rel_path, file)

            # Убираем './' в начале путей
            if file_path.startswith(f'.{os.sep}'):
                file_path = file_path[2:]

            f.write(f"{file_path}\n")

print(f"✓ Структура проекта сохранена в: {output_file}")
print(f"✓ Исключенные папки: {', '.join(exclude_dirs)}")
if exclude_hidden:
    print("✓ Скрытые файлы/папки исключены")
