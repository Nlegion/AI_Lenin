import os
import json
import torch
import gc
import traceback
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ======================
# КОНФИГУРАЦИЯ
# ======================
MODEL_DIR = "models/saiga"
ADAPTER_DIR = "models/saiga/lora_adapter"  # Новый адаптер
MERGED_MODEL_DIR = "models/saiga/merged_model"  # Новая папка для объединенной модели
os.makedirs(MERGED_MODEL_DIR, exist_ok=True)

# ======================
# ПРОВЕРКА ФАЙЛОВ
# ======================
print("Проверка файлов модели...")

# Проверяем наличие sharded модели
safetensors_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('model-') and f.endswith('.safetensors')]
index_file = os.path.join(MODEL_DIR, "model.safetensors.index.json")

if not safetensors_files:
    print("❌ Не найдены файлы модели .safetensors")
    exit(1)

if not os.path.exists(index_file):
    print("❌ Не найден index файл модели")
    exit(1)

print(f"✅ Найдено {len(safetensors_files)} sharded файлов модели")
print(f"✅ Найден index файл: {index_file}")

# Проверяем наличие адаптера
if not os.path.exists(ADAPTER_DIR):
    print(f"❌ Адаптер не найден: {ADAPTER_DIR}")
    exit(1)

adapter_files = os.listdir(ADAPTER_DIR)
print(f"Файлы адаптера: {adapter_files}")

# ======================
# ЗАГРУЗКА МОДЕЛИ И АДАПТЕРА
# ======================
try:
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=True,
        local_files_only=True,
        padding_side="right"
    )

    print("Загрузка базовой модели...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
        use_cache=False
    )

    print("✅ Базовая модель успешно загружена")

    print("Загрузка адаптера...")
    model = PeftModel.from_pretrained(
        model,
        ADAPTER_DIR,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("✅ Адаптер успешно загружен")

except Exception as e:
    print(f"❌ Ошибка загрузки: {str(e)}")
    traceback.print_exc()
    exit(1)

# ======================
# ОБЪЕДИНЕНИЕ МОДЕЛЕЙ
# ======================
try:
    print("🧩 Начало объединения моделей...")

    # Объединяем модель с адаптером
    merged_model = model.merge_and_unload()

    # Сохраняем объединенную модель
    merged_model.save_pretrained(
        MERGED_MODEL_DIR,
        safe_serialization=True
    )
    tokenizer.save_pretrained(MERGED_MODEL_DIR)

    print(f"💾 Объединенная модель сохранена в: {MERGED_MODEL_DIR}")

    # Сохраняем конфигурацию
    with open(os.path.join(MERGED_MODEL_DIR, "config.json"), "w") as f:
        json.dump(merged_model.config.to_dict(), f, indent=2)

    print("✅ Конфигурация сохранена")

except Exception as e:
    print(f"❌ Ошибка объединения: {str(e)}")
    traceback.print_exc()
    exit(1)

# ======================
# ПРОВЕРКА ОБЪЕДИНЕННОЙ МОДЕЛИ
# ======================
try:
    print("🔍 Проверка объединенной модели...")

    # Загружаем объединенную модель для проверки
    test_model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_DIR,
        device_map="cpu",
        torch_dtype=torch.float16,
        local_files_only=True
    )

    test_tokenizer = AutoTokenizer.from_pretrained(
        MERGED_MODEL_DIR,
        local_files_only=True
    )

    print("✅ Объединенная модель успешно загружена и проверена")

    # Освобождаем память
    del test_model
    del test_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

except Exception as e:
    print(f"❌ Ошибка проверки объединенной модели: {str(e)}")
    traceback.print_exc()
    exit(1)

print("\n🎉 Объединенная модель успешно создана!")
print("\nСЛЕДУЮЩИЕ ШАГИ:")
print(
    "1. Конвертация в GGUF: python llama.cpp/convert.py models/saiga/merged_model_new --outtype f16 --outfile models/saiga/lenin_merged_new.f16.gguf")
print(
    "2. Квантование: ./llama.cpp/quantize models/saiga/lenin_merged_new.f16.gguf models/saiga/lenin_merged_new.q4_k.gguf Q4_K_M")
print("3. Тестирование: python test_merged_model.py --model models/saiga/lenin_merged_new.q4_k.gguf")