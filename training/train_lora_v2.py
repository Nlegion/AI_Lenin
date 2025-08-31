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
# КОНФИГУРАЦИЯ (АДРЕСА)
# ======================
MODEL_DIR = "../models/saiga"
DATASET_PATH = "../data/finetune/lenin_lora_final.jsonl"
OUTPUT_DIR = "../models/saiga/lora_adapter"
MERGED_MODEL_DIR = "../models/saiga/merged_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_MODEL_DIR, exist_ok=True)

# ======================
# ПРОВЕРКА ФАЙЛОВ
# ======================
print("Проверка файлов модели...")
required_files = ["config.json", "model.safetensors.index.json", "tokenizer.json"]

for file in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, file)):
        print(f"❌ Файл не найден: {os.path.join(MODEL_DIR, file)}")
        exit(1)

# ======================
# ЗАГРУЗКА МОДЕЛИ (РЕШЕНИЕ ПРОБЛЕМЫ)
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
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_cache=False
    )
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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
# ПАРАМЕТРЫ ОБУЧЕНИЯ (ОКОНЧАТЕЛЬНОЕ РЕШЕНИЕ)
# ======================
use_bf16 = torch.cuda.is_bf16_supported()

# ОТКЛЮЧАЕМ ГРАДИЕНТНЫЙ ЧЕКПОИНТИНГ
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    gradient_checkpointing=False,  # ОТКЛЮЧЕНО ДЛЯ РЕШЕНИЯ ПРОБЛЕМЫ
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
# ОБУЧЕНИЕ (С ОТКЛЮЧЕННЫМ ГРАДИЕНТНЫМ ЧЕКПОИНТИНГОМ)
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
    model.train()
    trainer.train()
    print("🎉 Обучение завершено!")

    # Сохранение адаптера
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

except Exception as e:
    print(f"❌ Ошибка обучения: {str(e)}")
    traceback.print_exc()
    exit(1)

# ======================
# ОБЪЕДИНЕНИЕ МОДЕЛЕЙ (CPU)
# ======================
try:
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    print("🧩 Начало объединения моделей на CPU...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    lora_model = PeftModel.from_pretrained(
        base_model,
        OUTPUT_DIR,
        device_map="cpu"
    )

    merged_model = lora_model.merge_and_unload()

    merged_model.save_pretrained(MERGED_MODEL_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    print(f"💾 Объединенная модель сохранена в: {MERGED_MODEL_DIR}")

except Exception as e:
    print(f"❌ Ошибка объединения: {str(e)}")
    traceback.print_exc()

print("\nСЛЕДУЮЩИЕ ШАГИ:")
print(f"1. Конвертация в GGUF: python llama.cpp/convert.py {MERGED_MODEL_DIR}")
print("2. Квантование: ./llama.cpp/quantize lenin_model.f16.gguf lenin_model.Q4_K_M.gguf Q4_K_M")
print("3. Интеграция с llama.cpp для API")