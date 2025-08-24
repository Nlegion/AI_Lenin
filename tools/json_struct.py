import json
from pathlib import Path


def analyze_ontology_structure(ontology_path, output_path):
    """Анализирует структуру онтологии и сохраняет описание в файл"""
    ontology_path = Path(ontology_path)
    output_path = Path(output_path)

    if not ontology_path.exists():
        print(f"Ошибка: Файл онтологии не найден: {ontology_path}")
        return

    try:
        with open(ontology_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Ошибка при чтении файла онтологии: {str(e)}")
        return

    # Создаем отчет о структуре
    report_lines = [
        "Структура файла lenin_worldview_template.json",
        "=" * 50,
        f"Тип корневого элемента: {type(data).__name__}",
        ""
    ]

    if isinstance(data, dict):
        report_lines.append("Корневой элемент: словарь (dict)")
        report_lines.append(f"Количество концептов: {len(data)}")
        report_lines.append("\nСтруктура элементов:")

        for i, (concept, concept_data) in enumerate(data.items(), 1):
            report_lines.append(f"\nКонцепт #{i}: '{concept}'")
            report_lines.append(f"  Тип данных концепта: {type(concept_data).__name__}")

            if isinstance(concept_data, dict):
                report_lines.append("  Ключи в концепте:")
                for key, value in concept_data.items():
                    value_type = type(value).__name__
                    report_lines.append(f"    - '{key}': {value_type}")

                    # Для эмбеддингов показываем длину
                    if key == "embedding" and isinstance(value, list):
                        report_lines.append(f"        Длина эмбеддинга: {len(value)}")

                    # Для вложенных структур
                    if isinstance(value, (dict, list)):
                        report_lines.append(f"        Пример содержимого: {str(value)[:100]}...")

            elif isinstance(concept_data, list):
                report_lines.append(f"  Длина списка: {len(concept_data)}")
                if concept_data:
                    first_item = concept_data[0]
                    report_lines.append(f"  Тип первого элемента: {type(first_item).__name__}")

                    # Если это список чисел (эмбеддинг)
                    if all(isinstance(x, (int, float)) for x in concept_data):
                        report_lines.append("  Содержимое: числовой вектор (эмбеддинг)")
                        report_lines.append(f"  Длина вектора: {len(concept_data)}")
                    else:
                        report_lines.append(f"  Пример содержимого: {str(concept_data)[:100]}...")
            else:
                report_lines.append(f"  Значение: {str(concept_data)[:100]}...")

    elif isinstance(data, list):
        report_lines.append("Корневой элемент: список (list)")
        report_lines.append(f"Количество элементов: {len(data)}")

        if data:
            first_item = data[0]
            report_lines.append(f"\nТип первого элемента: {type(first_item).__name__}")

            if isinstance(first_item, dict):
                report_lines.append("Ключи в первом элементе:")
                for key in first_item.keys():
                    report_lines.append(f"  - '{key}'")

                # Проверяем наличие стандартных ключей
                if 'concept' in first_item:
                    report_lines.append("\nСтруктура элементов: список словарей с концептами")
                elif 'name' in first_item:
                    report_lines.append("\nСтруктура элементов: список словарей с концептами (ключ 'name')")

            # Анализ содержимого элементов
            for i, item in enumerate(data[:5], 1):  # Первые 5 элементов
                report_lines.append(f"\nЭлемент #{i}:")
                if isinstance(item, dict):
                    for key, value in item.items():
                        value_type = type(value).__name__
                        report_lines.append(f"  '{key}': {value_type}")

                        # Для эмбеддингов
                        if key == "embedding" and isinstance(value, list):
                            report_lines.append(f"      Длина эмбеддинга: {len(value)}")
                else:
                    report_lines.append(f"  Тип: {type(item).__name__}")
                    report_lines.append(f"  Значение: {str(item)[:100]}...")

    # Сохраняем отчет в файл
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"Отчет о структуре сохранен в: {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении отчета: {str(e)}")


if __name__ == "__main__":
    # Конфигурация путей
    PROJECT_ROOT = Path(__file__).parent
    ONTOLOGY_PATH = PROJECT_ROOT / "lenin_worldview_template.json"
    OUTPUT_PATH = PROJECT_ROOT / "lenin_worldview_template.txt"

    # Анализ структуры
    analyze_ontology_structure(ONTOLOGY_PATH, OUTPUT_PATH)