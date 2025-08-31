import os
from sentence_transformers import SentenceTransformer

# Путь для сохранения модели
MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_PATH = os.path.join("../models", "sentence-transformers", MODEL_NAME)

# Создаем директорию, если её нет
os.makedirs(SAVE_PATH, exist_ok=True)

print(f"Загрузка модели {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print(f"Сохранение модели в {SAVE_PATH}...")
model.save(SAVE_PATH)

print("Модель успешно сохранена!")