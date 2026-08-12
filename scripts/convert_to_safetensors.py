from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.core.settings.config import Settings

token = Settings().HUGGINGFACE_TOKEN
MODEL_ID = "IlyaGusev/saiga_mistral_7b"
MODEL_REVISION = "main"

# Explicitly request safetensors serialization on save.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16,
    token=token,
    revision=MODEL_REVISION,
)

model.save_pretrained(
    "./models/saiga_mistral_7b_safe",
    safe_serialization=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

input_text = "Пролетарии всех стран, соединяйтесь!"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
