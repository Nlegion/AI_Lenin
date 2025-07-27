import torch
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from src.core.settings.config import Settings
import logging
import html
import gc

logger = logging.getLogger(__name__)


class LeninAnalyzer:
    def __init__(self, vector_db_path: str = "database/vector_db"):
        self.config = Settings()
        token = self.config.HUGGINGFACE_TOKEN

        # Оптимизированная конфигурация для 8GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # Используем float16 вместо bfloat16
            bnb_4bit_quant_type="nf4",  # Новый тип 4-битного квантования
            bnb_4bit_use_double_quant=True,
        )

        # Инициализация токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME,
            use_fast=False,
            token=token,
            legacy=False
        )

        # Загрузка модели с оптимизациями
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,  # Используем float16 для экономии памяти
            token=token,
            attn_implementation="flash_attention_2",  # Используем FlashAttention
            low_cpu_mem_usage=True  # Минимизируем использование CPU RAM
        )
        logger.info("Модель инициализирована")
        # Инициализация векторной БД на CPU
        self._init_vector_db(vector_db_path)

        logger.info("Векторная БД инициализирована")

    def _init_vector_db(self, vector_db_path: str):
        # Используем CPU для векторной БД
        self.embed_model = SentenceTransformer(
            self.config.EMBEDDING_MODEL,
            device='cpu'  # Используем CPU вместо CUDA
        )

        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)

        try:
            self.collection = self.chroma_client.get_collection(
                name="lenin_works",
                embedding_function=lambda texts: self.embed_model.encode(texts).tolist()
            )
        except ValueError:
            self.collection = self.chroma_client.create_collection(
                name="lenin_works",
                embedding_function=lambda texts: self.embed_model.encode(texts).tolist()
            )

    def retrieve_context(self, query: str, k: int = 3) -> str:  # Уменьшаем количество контекста
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents"]
            )
            if 'documents' in results and len(results['documents']) > 0:
                return "\n\n".join(results['documents'][0])
            return ""
        except Exception as e:
            logger.error(f"Ошибка поиска контекста: {str(e)}")
            return ""

    def generate_analysis(self, news_item) -> str:
        try:
            query = f"{news_item.title} {news_item.content[:300]}"  # Уменьшаем длину запроса
            context = self.retrieve_context(query)

            prompt = f"""
            <s>[INST]Ты — Владимир Ильич Ленин. Кратко проанализируй новость с позиции классовой борьбы:

            Новость:
            {html.escape(news_item.title)}
            {html.escape(news_item.content[:1000])}

            Контекст:
            {context}[/INST]
            Анализ:
            """

            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

            # Генерация с оптимизацией памяти
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,  # Уменьшаем количество токенов
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Очистка памяти сразу после генерации
            del inputs
            torch.cuda.empty_cache()
            gc.collect()

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Анализ:")[-1].strip()
        except Exception as e:
            logger.error(f"Ошибка генерации анализа: {str(e)}")
            return "Не удалось сгенерировать анализ."