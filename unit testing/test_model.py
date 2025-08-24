from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import safetensors.torch
from src.core.settings.config import Settings
token = Settings().HUGGINGFACE_TOKEN

# Явно укажем использовать safetensors
model = AutoModelForCausalLM.from_pretrained(
    "IlyaGusev/saiga_mistral_7b",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16,
            token=token
)

model.save_pretrained(
    "./models/saiga_mistral_7b_safe",
    safe_serialization=True
)

tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/saiga_mistral_7b")

input_text = "Пролетарии всех стран, соединяйтесь!"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))