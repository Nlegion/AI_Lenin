import logging
import re
import os
import gc
import torch
import chromadb
import numpy as np
from llama_cpp import Llama
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb import PersistentClient
from functools import lru_cache
from src.core.settings.config import Settings

logger = logging.getLogger(__name__)


class LeninAnalyzer:
    def __init__(self, vector_db_path: str = None):
        logger.info("Инициализация LeninAnalyzer")
        self.config = Settings()

        # Определение абсолютных путей
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(BASE_DIR, "models", "saiga")

        self.model_config = {
            "model_path": os.path.join(model_dir, "model-q4_K.gguf"),
            "lora_path": os.path.join(model_dir, "lora_adapter.gguf"),
            "embedding_model": "all-MiniLM-L6-v2"
        }

        # Путь к векторной БД по умолчанию
        if vector_db_path is None:
            vector_db_path = os.path.join(BASE_DIR, "database", "vector_db")

        self.llm = None
        self.embedding_function = None
        self._init_vector_db(vector_db_path)
        logger.info(f"Векторная БД инициализирована: {vector_db_path}")
        self.reload_model()

    def reload_model(self):
        try:
            if self.llm is not None:
                logger.info("Выгрузка предыдущей модели...")
                del self.llm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Очищена память CUDA")

            logger.info("Загрузка Saiga-Llama3-8b с LoRA адаптером")
            n_gpu_layers = 35 if torch.cuda.is_available() else 0
            n_threads = 8

            self.llm = Llama(
                model_path=self.model_config["model_path"],
                lora_path=self.model_config["lora_path"] if os.path.exists(self.model_config["lora_path"]) else None,
                n_ctx=4096,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format="llama-3",
                verbose=False,
                n_batch=256,
                use_mlock=True
            )
            logger.info(f"Модель загружена (GPU слои: {n_gpu_layers}, потоки: {n_threads})")
        except Exception as e:
            logger.exception(f"Ошибка загрузки модели: {str(e)}")
            logger.warning("Попытка загрузки в безопасном режиме...")
            self.llm = Llama(
                model_path=self.model_config["model_path"],
                n_ctx=2048,
                n_gpu_layers=0,
                n_threads=4,
                chat_format="llama-3",
                verbose=False
            )

    def _init_vector_db(self, path: str):
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.model_config["embedding_model"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.chroma_client = PersistentClient(path=path)
        try:
            self.collection = self.chroma_client.get_collection(
                name="lenin_works",
                embedding_function=self.embedding_function
            )
            logger.info("Коллекция lenin_works успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки коллекции: {str(e)}")
            self.collection = None

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
        text = re.sub(
            r'(\[INST\].*?\[/INST\]|<\/?s>|Анализ Ленина:|```|SYSTEM:|USER:|ASSISTANT:|\([^)]*\)|"|«|»)',
            '', text
        )
        text = re.sub(r'Владимир Ильич (Ленин)?[:,]?\s*', '', text, flags=re.IGNORECASE)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        return ' '.join(sentences[:2]).strip()

    def generate_analysis(self, news_title: str, news_content: str) -> str:
        try:
            if self.llm is None:
                self.reload_model()

            query = f"{news_title} {news_content[:100]}"
            context = self.retrieve_context(query)

            system_prompt = (
                "Ты — Владимир Ильич Ленин. Дай марксистский анализ классовой борьбы ровно в 2 предложениях. "
                "Используй революционный стиль без упоминания стран и национальностей. "
                "Избегай абстрактных рассуждений\n\n"
                # "В ответе используй термины: пролетариат, буржуазия, революция, эксплуатация."
            )

            user_content = f"Контекст: {context}\n\nНовость: {news_title}\n{news_content[:500]}" if context else \
                f"Новость: {news_title}\n{news_content[:500]}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                top_p=0.85,
                max_tokens=150,
                stop=["<|eot_id|>", "\n\n"],
                repeat_penalty=1.1
            )

            analysis = response['choices'][0]['message']['content'].strip()
            return self.clean_analysis(analysis)

        except Exception as e:
            logger.exception(f"Ошибка генерации: {str(e)}")
            return "Не удалось сгенерировать анализ."