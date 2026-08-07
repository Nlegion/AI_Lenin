import os
from sentence_transformers import SentenceTransformer

# Путь для сохранения модели
MODEL_NAME = "ai-sage/Giga-Embeddings-instruct"
SAVE_PATH = os.path.join("../models", "Giga-Embeddings-instruct")

# Создаем директорию, если её нет
os.makedirs(SAVE_PATH, exist_ok=True)

print(f"Загрузка модели {MODEL_NAME}...")
model = SentenceTransformer(model_name_or_path=MODEL_NAME, trust_remote_code=True)

print(f"Сохранение модели в {SAVE_PATH}...")
model.save(SAVE_PATH)

print("Модель успешно сохранена!")