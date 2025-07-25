import os
import re
from sentence_transformers import SentenceTransformer


def clean_text(text):
    """Очистка текстов Ленина от специфичных артефактов"""
    text = re.sub(r'\n+', ' ', text)  # Удаляем множественные переносы
    text = re.sub(r'\s+', ' ', text)  # Заменяем множественные пробелы
    text = re.sub(r'\[\d+\]', '', text)  # Удаляем сноски типа [123]
    return text.strip()


def prepare_dataset():
    """Загрузка и подготовка текстов"""
    texts = []
    metadata = []

    # Пример структуры папки: data/raw/том1.txt, том2.txt...
    for filename in os.listdir('data/raw'):
        if filename.endswith('.txt'):
            with open(f'data/raw/{filename}', 'r', encoding='utf-8') as f:
                content = f.read()

                # Разбиваем на главы/разделы
                sections = re.split(r'(?=\nГЛАВА|\nРАЗДЕЛ|\n\d+\. )', content)
                for i, section in enumerate(sections):
                    if len(section) < 50:  # Пропускаем слишком короткие
                        continue
                    texts.append(clean_text(section))
                    metadata.append({
                        'source': filename,
                        'section': i + 1,
                        'length': len(section)
                    })

    return texts, metadata


if __name__ == "__main__":
    texts, metadata = prepare_dataset()
    print(f"Подготовлено {len(texts)} текстовых сегментов")