import os
import json
import torch
import traceback
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset
from trl import SFTTrainer

# ======================
# КОНФИГУРАЦИЯ
# ======================
MODEL_DIR = "models/saiga"  # Путь к локальной модели
DATASET_PATH = "data/finetune/lenin_lora_final.jsonl"
OUTPUT_DIR = "models/saiga/lora_adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# ПРОВЕРКА ФАЙЛОВ
# ======================
print("Проверка файлов модели...")

# Критически важные файлы
required_files = [
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "special_tokens_map.json"
]

for file in required_files:
    path = os.path.join(MODEL_DIR, file)
    if not os.path.exists(path):
        print(f"❌ Файл не найден: {path}")
        exit(1)

# Проверка шардов модели
index_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
with open(index_path, "r") as f:
    index_data = json.load(f)

shard_files = set(index_data["weight_map"].values())

for shard in shard_files:
    path = os.path.join(MODEL_DIR, shard)
    if not os.path.exists(path):
        print(f"❌ Шард модели не найден: {path}")
        exit(1)

print("✅ Все файлы модели найдены")

# ======================
# ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА
# ======================
print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=True,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

# Для формата чата Llama3
tokenizer.add_special_tokens({
    "eos_token": "<|eot_id|>",
    "bos_token": "<|begin_of_text|>"
})

print("Загрузка модели...")
try:
    # Конфигурация для 4-битной загрузки
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    print("✅ Модель успешно загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {str(e)}")
    traceback.print_exc()
    exit(1)

# ======================
# НАСТРОЙКА LoRA
# ======================
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ======================
# ПОДГОТОВКА ДАННЫХ
# ======================
print("Загрузка датасета...")
try:
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"✅ Загружено {len(dataset)} примеров")


    # Функция для преобразования формата разговора в текст
    def format_conversation(example):
        conversation = example["conversations"]
        text = ""

        # Специальные токены для формата чата Llama3
        BOS = "<|begin_of_text|>"
        EOS = "<|eot_id|>"
        SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>"
        USER_HEADER = "<|start_header_id|>user<|end_header_id|>"
        ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"

        for message in conversation:
            role = message["role"]
            content = message["content"].strip()

            if role == "system":
                text += f"{BOS}{SYSTEM_HEADER}\n\n{content}{EOS}"
            elif role == "user":
                text += f"{USER_HEADER}\n\n{content}{EOS}"
            elif role == "assistant":
                text += f"{ASSISTANT_HEADER}\n\n{content}{EOS}"

        return {"text": text}


    # Применяем преобразование ко всему датасету
    print("Преобразование формата датасета...")
    dataset = dataset.map(format_conversation)

    # Проверка результата
    sample_text = dataset[0]["text"]
    print(f"Пример преобразованного текста:\n{sample_text[:200]}...")

except Exception as e:
    print(f"❌ Ошибка загрузки датасета: {str(e)}")
    traceback.print_exc()
    exit(1)

# ======================
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    optim="paged_adamw_8bit",
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# ======================
# СОЗДАНИЕ ТРЕНЕРА
# ======================
print("Создание тренера...")
try:
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False
    )
    print("✅ Тренер успешно создан")
except Exception as e:
    print(f"❌ Ошибка создания тренера: {str(e)}")
    traceback.print_exc()
    exit(1)

# ======================
# ОБУЧЕНИЕ
# ======================
print("🚀 Начало обучения LoRA...")
try:
    trainer.train()
    print("🎉 Обучение завершено успешно!")
except Exception as e:
    print(f"❌ Ошибка во время обучения: {str(e)}")
    traceback.print_exc()
    exit(1)

# ======================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ======================
try:
    # Сохраняем LoRA адаптер
    model.save_pretrained(OUTPUT_DIR)
    print(f"💾 LoRA адаптер сохранён в: {OUTPUT_DIR}")

    # Сохраняем токенизатор
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Токенизатор сохранён в: {OUTPUT_DIR}")

    print(f"💾 Конфигурация обучения сохранена")
except Exception as e:
    print(f"❌ Ошибка сохранения модели: {str(e)}")
    traceback.print_exc()