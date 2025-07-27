import requests

model_id = "IlyaGusev/saiga2_7b_lora"
response = requests.get(f"https://huggingface.co/api/models/{model_id}")

if response.status_code == 401:
    print("Требуется согласие на использование модели")
    print("Посетите: https://huggingface.co/IlyaGusev/saiga2_7b_lora")
elif response.status_code == 200:
    print("Модель доступна")
else:
    print(f"Статус: {response.status_code}")