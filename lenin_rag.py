import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from chromadb import PersistentClient  # Или from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Конфигурация
EMBED_MODEL = 'intfloat/multilingual-e5-large'
LLM_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'


class LeninAI:
    def __init__(self):
        # Инициализация модели эмбеддингов
        self.embed_model = SentenceTransformer(EMBED_MODEL, device='cuda')

        # Инициализация векторной базы (Chroma пример)
        self.client = PersistentClient(path="lenin_chroma_db")
        self.collection = self.client.get_collection("lenin_works")

        # Инициализация LLM
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7
        )

    def retrieve_documents(self, query: str, k: int = 3):
        """Поиск релевантных документов"""
        query_embed = self.embed_model.encode([query])
        results = self.collection.query(
            query_embeddings=[query_embed.tolist()],
            n_results=k
        )
        return results['documents'][0]

    def generate_response(self, query: str):
        """Генерация ответа в стиле Ленина"""
        # Поиск релевантных контекстов
        context_docs = self.retrieve_documents(query)
        context = "\n\n".join(context_docs)

        # Формирование промпта
        prompt = f"""Ты — Владимир Ильич Ленин. Отвечай на вопрос, используя свой характерный стиль и идеологию.

Контекст:
{context}

Вопрос: {query}

Ответ:
"""
        # Генерация ответа
        response = self.pipe(
            prompt,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]['generated_text']

        # Извлекаем только ответ после промпта
        return response.split("Ответ:")[-1].strip()


if __name__ == "__main__":
    ai = LeninAI()

    print("Система готова. Задайте вопрос Ленину (exit для выхода)")
    while True:
        query = input("\nВаш вопрос: ")
        if query.lower() == 'exit':
            break

        response = ai.generate_response(query)
        print(f"\nЛенин: {response}")