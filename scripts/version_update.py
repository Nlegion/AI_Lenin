#!/usr/bin/env python3
import argparse
from pathlib import Path


def update_version(version_type):
    """Обновление версии проекта"""
    version_file = Path("VERSION")

    # Чтение текущей версии
    if version_file.exists():
        with open(version_file, 'r') as f:
            current_version = f.read().strip()
    else:
        current_version = "1.0.0"

    # Разбор версии
    parts = current_version.split('.')
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    # Обновление версии
    if version_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif version_type == "minor":
        minor += 1
        patch = 0
    elif version_type == "patch":
        patch += 1

    new_version = f"{major}.{minor}.{patch}"

    # Запись новой версии
    with open(version_file, 'w') as f:
        f.write(new_version)

    print(f"Версия обновлена: {current_version} -> {new_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обновление версии проекта")
    parser.add_argument("type", choices=["major", "minor", "patch"],
                        help="Тип обновления версии")

    args = parser.parse_args()
    update_version(args.type)