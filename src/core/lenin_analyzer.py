import logging
import re
import os
import aiohttp
import torch
import chromadb
import numpy as np
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb import PersistentClient
from functools import lru_cache
from src.core.settings.config import Settings

logger = logging.getLogger(__name__)


class LeninAnalyzer:
    def __init__(self, vector_db_path: str = None):
        logger.info("Инициализация LeninAnalyzer (серверный режим)")
        self.config = Settings()
        self.server_url = "http://127.0.0.1:8080"
        self.session = None

        # Определение абсолютных путей
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Инициализация векторной БД
        if vector_db_path is None:
            vector_db_path = os.path.join(BASE_DIR, "database", "vector_db")

        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.chroma_client = PersistentClient(path=vector_db_path)
        try:
            self.collection = self.chroma_client.get_collection(
                name="lenin_works",
                embedding_function=self.embedding_function
            )
            logger.info("Коллекция lenin_works успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки коллекции: {str(e)}")
            self.collection = None

    async def initialize_session(self):
        """Инициализация HTTP сессии"""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))

    async def close_session(self):
        """Закрытие HTTP сессии"""
        if self.session:
            await self.session.close()
            self.session = None

    @lru_cache(maxsize=500)
    def cached_embedding(self, text: str) -> list:
        return self.embedding_function([text])[0]

    def retrieve_context(self, query: str, k: int = 3) -> str:
        try:
            if not self.collection or self.collection.count() == 0:
                return ""

            embedding = self.cached_embedding(query)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["documents"]
            )

            context = " ".join(
                doc for doc_list in results['documents']
                for doc in doc_list if isinstance(doc, str)
            )[:500]

            return context if context else ""
        except Exception as e:
            logger.exception(f"Ошибка поиска контекста: {str(e)}")
            return ""

    def clean_analysis(self, text: str) -> str:
        # Жесткое удаление шаблонных фраз
        forbidden_patterns = [
            r'Теперь[^.!?]*[.!?]',
            r'Рассмотрим[^.!?]*[.!?]',
            r'Анализируя[^.!?]*[.!?]',
            r'можно сделать вывод[^.!?]*[.!?]',
            r'данная ситуация[^.!?]*[.!?]',
            r'В контексте[^.!?]*[.!?]'
        ]

        for pattern in forbidden_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Разбиваем на предложения и берем только первые два законченных
        sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in '.!?':
                sentences.append(current_sentence.strip())
                current_sentence = ""
                if len(sentences) >= 2:
                    break

        # Если набрали меньше двух предложений, добавляем оставшийся текст
        if len(sentences) < 2 and current_sentence:
            sentences.append(current_sentence.strip())

        return ' '.join(sentences[:2])

    async def generate_analysis(self, news_title: str, news_content: str) -> str:
        try:
            await self.initialize_session()

            query = f"{news_title} {news_content[:100]}"
            context = self.retrieve_context(query)

            system_prompt = (
                "Дай краткий анализ новости с точки зрения марксиста.\n\n"
                "Ответ должен быть строго в 2 предложения, конкретный и основанный только на фактах из новости."
                "Запрещены: вводные фразы, упоминание стран, революции, военных аспектов.\n\n"
                "Анализируй только: труд, политика, капитал, ресурсы, классовые противоречия."
            )

            user_content = f"Контекст: {context}\n\nНовость: {news_title}\n{news_content[:500]}" if context else \
                f"Новость: {news_title}\n{news_content[:500]}"

            # Форматируем запрос для llama.cpp сервера
            prompt = self._format_llama3_prompt(system_prompt, user_content)

            data = {
                "prompt": prompt,
                "n_predict": 80,
                "temperature": 0.3,
                "top_p": 0.5,
                "repeat_penalty": 1.4,
                "stop": ["\n", "###", "Анализ:", "Ленин:", "Теперь", "Рассмотрим", "Новость"]
            }

            async with self.session.post(
                    f"{self.server_url}/completion",
                    json=data,
                    headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get('content', '').strip()
                    return self.clean_analysis(content)
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка сервера: {response.status} - {error_text}")
                    return "Не удалось сгенерировать анализ."

        except Exception as e:
            logger.exception(f"Ошибка генерации: {str(e)}")
            return "Не удалось сгенерировать анализ."

    def _format_llama3_prompt(self, system_prompt: str, user_input: str) -> str:
        """Форматирует промпт для модели Llama 3"""
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_input}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )