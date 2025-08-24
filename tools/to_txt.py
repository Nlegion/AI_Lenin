import os

output_file = "project.txt"
included_extensions = {"py", "html", "ini", "js", "css",  "yaml", "yml", "sh", "md"}
excluded_dirs = {".venv", "tmp", "venv", "logs", "data", "models"}

def should_exclude(path):
    for excluded_dir in excluded_dirs:
        if excluded_dir in path.split(os.sep):
            return True
    return False

def process_files():
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("")

    print("Начинаем сборку файлов проекта...")
    print(f"Включаемые расширения: {', '.join(included_extensions)}")
    print(f"Исключаемые директории: {', '.join(excluded_dirs)}")
    print()

    files_to_process = []
    for root, dirs, files in os.walk(".."):
        if should_exclude(root):
            continue
        for file in files:
            ext = file.split(".")[-1]
            if ext in included_extensions:
                files_to_process.append(os.path.join(root, file))

    total_files = len(files_to_process)
    processed_files = 0

    for file_path in files_to_process:
        processed_files += 1
        print(f"Обрабатывается файл {processed_files}/{total_files}: {file_path}", end="\r")

        if file_path != output_file and not should_exclude(file_path):
            with open(output_file, "a", encoding="utf-8") as outfile:
                outfile.write(f"=== {file_path} ===\n")
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                except UnicodeDecodeError:
                    try:
                        with open(file_path, "r", encoding="latin-1") as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        print(f"\nНе удалось прочитать файл {file_path}: {e}")
                outfile.write("\n\n")
        else:
            print(f"Пропущен файл или директория: {file_path}")

    print(f"\nГотово! Содержимое файлов сохранено в {output_file}")

if __name__ == "__main__":
    process_files()
