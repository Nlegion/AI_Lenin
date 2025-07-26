import torch
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from src.core.settings.config import Settings
import logging
import html

logger = logging.getLogger(__name__)


class LeninAnalyzer:
    def __init__(self, vector_db_path: str = "database/vector_db"):
        self.config = Settings()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME,
            use_fast=False
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        # Инициализация векторной БД (ChromaDB)
        self.embed_model = SentenceTransformer(
            self.config.EMBEDDING_MODEL,
            device='cuda'
        )
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="lenin_works",
            embedding_function=self.embed_model.encode
        )

    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Получение контекста из работ Ленина"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents"]
            )
            return "\n\n".join(results['documents'][0])
        except Exception as e:
            logger.error(f"Ошибка поиска контекста: {str(e)}")
            return ""

    def generate_analysis(self, news_item) -> str:
        """Генерация анализа на основе новости"""
        try:
            query = f"{news_item.title} {news_item.content[:500]}"
            context = self.retrieve_context(query)

            prompt = f"""
            <s>Ты — Владимир Ильич Ленин. Проанализируй новость с позиции классовой борьбы:

            Новость:
            Заголовок: {html.escape(news_item.title)}
            Текст: {html.escape(news_item.content[:2000])}

            Контекст из твоих работ:
            {context}

            Анализ:
            """

            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Анализ:")[-1].strip()
        except Exception as e:
            logger.error(f"Ошибка генерации анализа: {str(e)}")
            return "Не удалось сгенерировать анализ."