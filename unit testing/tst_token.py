import requests
import os
from src.core.settings.config import Settings

token = Settings().HUGGINGFACE_TOKEN
print("Token:", token)

headers = {"Authorization": f"Bearer {token}"}
response = requests.get("https://huggingface.co/api/whoami", headers=headers)

print("Status code:", response.status_code)
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.text)