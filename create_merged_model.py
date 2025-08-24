import os
import json
import torch
import gc
import traceback
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from trl import SFTTrainer

# ======================
# КОНФИГУРАЦИЯ
# ======================
MODEL_DIR = "models/saiga"
DATASET_PATH = "data/finetune/lenin_lora_final.jsonl"
OUTPUT_DIR = "models/saiga/lora_adapter"  # Новая папка для адаптера
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# ПРОВЕРКА ФАЙЛОВ МОДЕЛИ
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

# ======================
# ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА
# ======================
try:
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=True,
        local_files_only=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        "eos_token": "<|eot_id|>",
        "bos_token": "<|begin_of_text|>"
    })

    print("Загрузка модели с 4-bit квантованием...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_cache=False
    )

    print("✅ Модель успешно загружена")

except Exception as e:
    print(f"❌ Ошибка загрузки: {str(e)}")
    traceback.print_exc()
    exit(1)

# ======================
# НАСТРОЙКА LoRA
# ======================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ======================
# ПОДГОТОВКА ДАННЫХ
# ======================
try:
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"✅ Загружено {len(dataset)} примеров")


    def format_conversation(example):
        BOS = "<|begin_of_text|>"
        EOS = "<|eot_id|>"
        formatted = BOS

        for msg in example["conversations"]:
            role = msg["role"]
            content = msg["content"].strip()
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}{EOS}"

        return formatted


    dataset = dataset.map(
        lambda x: {"text": format_conversation(x)},
        remove_columns=["conversations"]
    )
    print(f"Пример текста: {dataset[0]['text'][:200]}...")

except Exception as e:
    print(f"❌ Ошибка датасета: {str(e)}")
    traceback.print_exc()
    exit(1)

# ======================
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ======================
use_bf16 = torch.cuda.is_bf16_supported()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_steps=50,  # Сохраняем каждые 50 шагов
    save_total_limit=3,  # Храним только 3 последних чекпоинта
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    gradient_checkpointing=False,
    group_by_length=True,
    bf16=use_bf16,
    fp16=not use_bf16,
    report_to="none",
    remove_unused_columns=True,
    label_names=["input_ids", "attention_mask"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ======================
# ОБУЧЕНИЕ
# ======================
try:
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=False,
        data_collator=data_collator,
        neftune_noise_alpha=5,
    )

    print("🚀 Начало обучения...")
    model.config.use_cache = False
    trainer.train()
    print("🎉 Обучение завершено!")

    # Сохранение финального адаптера
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Создаем README с информацией об адаптере
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
        f.write("# Адаптер Ленина для Saiga3\n\n")
        f.write("## Параметры обучения\n")
        f.write(f"- Модель: {MODEL_DIR}\n")
        f.write(f"- Датасет: {DATASET_PATH}\n")
        f.write(f"- Эпохи: 3\n")
        f.write(f"- Learning rate: 2e-5\n")
        f.write(f"- LoRA r: 16\n")
        f.write(f"- LoRA alpha: 32\n")

    print(f"💾 Адаптер сохранен в: {OUTPUT_DIR}")

except Exception as e:
    print(f"❌ Ошибка обучения: {str(e)}")
    traceback.print_exc()
    exit(1)

print("\n🎉 Адаптер успешно создан!")
print("\nСЛЕДУЮЩИЕ ШАГИ:")
print("1. Объединение модели с адаптером: python merge_model.py")
print("2. Конвертация в GGUF")
print("3. Квантование")