from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re

# Очистка памяти перед загрузкой
torch.cuda.empty_cache()

model_path = "../models/saiga2_7b"

print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

print("Загрузка модели с 4-битным квантованием...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        max_memory={0: "7GB"},
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True
        }
    )
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("Обнаружена нехватка VRAM! Загружаем базовую версию...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        raise

print("Создание пайплайна...")
lenin_analyzer = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

bad_words_ids = [
        [tokenizer.convert_tokens_to_ids("[INST]")],
        [tokenizer.convert_tokens_to_ids("[/INST]")],
        [tokenizer.convert_tokens_to_ids("<s>")],
        [tokenizer.convert_tokens_to_ids("</s>")]
    ]

prompt = """<s>[INST]
Ты — Владимир Ильич Ленин. Дай марксистский анализ с классовой позиции.
Строго соблюдай:
1. НИКОГДА не пересказывай вопрос
2. Только классовый анализ
3. Ровно 2 предложения

Запрещено:
- Упоминать страны или национальности
- Использовать слова из вопроса
- Пассивные конструкции

Анализируй суть:
Военные новости: Россия усиливает военную мощь.

Твой анализ:[/INST]
"""

print("Генерация анализа...")
generation_config = {
    "max_new_tokens": 70,
    "min_new_tokens": 15,
    "temperature": 0.92,
    "top_p": 0.87,
    "top_k": 40,
    "typical_p": 0.9,
    "repetition_penalty": 1.7,
    "encoder_repetition_penalty": 1.5,
    "do_sample": True,  # Возвращаем сэмплирование
    "num_beams": 3,
    "early_stopping": True,
    "length_penalty": -0.5,
    "no_repeat_ngram_size": 2,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.eos_token_id,
    "forced_eos_token_id": tokenizer.eos_token_id,
    "bad_words_ids": bad_words_ids,
    "renormalize_logits": True,
    "use_cache": False
}

result = lenin_analyzer(prompt, **generation_config)


def clean_output(text: str) -> str:
    # Удаление тегов
    tags = ["[INST]", "[/INST]", "<s>", "</s>"]
    for tag in tags:
        text = text.replace(tag, "")

    # Удаление упоминаний Ленина в третьем лице
    text = re.sub(r"(Владимир\s+Ильич\s+Ленин\s*[:,]?\s*)", "", text)

    # Обрезка до последней точки
    if '.' in text:
        last_period = text.rfind('.')
        text = text[:last_period + 1]

    # Удаление лишних пробелов и переносов
    text = " ".join(text.split()).strip()

    # Убедимся, что текст начинается с заглавной буквы
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


output = result[0]['generated_text']
analysis = clean_output(output.split("[/INST]")[-1])

print("\n=== Революционный анализ ===")
print(analysis)