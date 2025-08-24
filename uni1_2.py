import json
import numpy as np


def combine_stage_data(ontology_path, specificity_path, output_path):
    # Загрузка данных этапа 1
    with open(ontology_path, 'r', encoding='utf-8') as f:
        ontology = json.load(f)

    # Загрузка данных этапа 2
    with open(specificity_path, 'r', encoding='utf-8') as f:
        specificity = json.load(f)

    # Создаем объединенную структуру
    combined = {
        "transformations": {},
        "interpretations": {},
        "relations": ontology["relations"],
        "embeddings": {}
    }

    # Сопоставляем концепты
    all_concepts = set(ontology["concepts"])
    analyzed_concepts = set(specificity["interpretations"].keys())

    # Добавляем данные из этапа 2
    for concept in analyzed_concepts:
        if concept in specificity["transformations"]:
            combined["transformations"][concept] = {
                "transformation_index": specificity["transformations"][concept]["transformation_index"],
                "occurrences": specificity["transformations"][concept]["occurrences"]
            }

            combined["interpretations"][concept] = {
                "original_embedding": specificity["interpretations"][concept]["original_embedding"]
            }

    # Добавляем недостающие концепты из этапа 1
    for concept in all_concepts - analyzed_concepts:
        if concept in ontology["embeddings"]:
            combined["embeddings"][concept] = ontology["embeddings"][concept]

            # Для непроанализированных концептов используем нейтральные значения
            combined["transformations"][concept] = {
                "transformation_index": 1.0,
                "occurrences": 0
            }

            combined["interpretations"][concept] = {
                "original_embedding": ontology["embeddings"][concept]
            }

    # Сохраняем объединенные данные
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Объединенные данные сохранены в {output_path}")
    print(f"Всего концептов: {len(all_concepts)}")
    print(f"Проанализированных концептов: {len(analyzed_concepts)}")
    print(f"Добавленных концептов: {len(all_concepts - analyzed_concepts)}")


# Пример использования
combine_stage_data(
    ontology_path="data/stage1/foundation_ontology.json",
    specificity_path="data/stage2/lenin_dialectical_specificity_top1000.json",
    output_path="data/stage3/combined_worldview_data.json"
)