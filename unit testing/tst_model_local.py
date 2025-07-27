from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

model_path = "../models/saiga2_7b_lora"

# Конфигурация для 4-битной загрузки
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Используем float16 для вычислений
    bnb_4bit_quant_type="nf4",             # Тип квантования
    bnb_4bit_use_double_quant=True,        # Двойное квантование для экономии памяти
)

# Загрузка конфигурации адаптера
config = PeftConfig.from_pretrained(model_path)

# Загрузка базовой модели с квантованием
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# Загрузка адаптеров
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Генерация текста с улучшенными параметрами
input_text = "Пролетарии всех стран, соединяйтесь!"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Параметры генерации для более стабильного вывода
outputs = model.generate(
    **inputs,
    max_new_tokens=200,  # Увеличим количество новых токенов
    temperature=0.7,     # Контроль случайности
    top_p=0.9,           # Выбор из наиболее вероятных вариантов
    repetition_penalty=1.2, # Штраф за повторения
    do_sample=True       # Включить случайную выборку
)

# Декодируем с пропуском специальных токенов
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)