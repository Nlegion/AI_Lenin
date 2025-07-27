import torch
import os
from peft import PeftModel, PeftConfig
import threading
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from src.core.settings.config import Settings
import logging
import html
import gc

logger = logging.getLogger(__name__)


class LeninAnalyzer:
    def __init__(self, vector_db_path: str = "database/vector_db"):
        logger.info("Начало инициализации LeninAnalyzer")
        logger.info(f"PID: {os.getpid()}, Thread: {threading.get_ident()}")

        self.config = Settings()

        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        except Exception as e:
            logger.error(f"Error checking CUDA: {str(e)}")

        # Оптимизированная конфигурация для 8GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Инициализация токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME,
            use_fast=False,
            legacy=False
        )

        # Загрузка модели с оптимизациями
        try:
            logger.info("Загрузка модели с адаптерами...")

            # Загрузка конфигурации PEFT
            peft_config = PeftConfig.from_pretrained(self.config.MODEL_NAME)

            # Указываем локальный путь к базовой модели
            base_model_path = "models/Llama-2-7B-fp16"

            # Загрузка базовой модели
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            # Применение адаптеров
            self.model = PeftModel.from_pretrained(
                base_model,
                self.config.MODEL_NAME
            )
            self.model.eval()

            logger.info("Модель успешно загружена с локального пути")
        except Exception as e:
            logger.exception(f"Ошибка загрузки модели: {str(e)}")
            raise


        logger.info("Модель инициализирована")
        # Инициализация векторной БД на CPU
        self._init_vector_db(vector_db_path)

        logger.info("Векторная БД инициализирована")

    def _init_vector_db(self, vector_db_path: str):
        logger.info(f"Инициализация векторной БД по пути: {os.path.abspath(vector_db_path)}")

        # Используем встроенный метод ChromaDB
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.config.EMBEDDING_MODEL,
            device="cpu"
        )

        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)

        try:
            self.collection = self.chroma_client.get_collection(
                name="lenin_works",
                embedding_function=embedding_function
            )
            logger.info("Коллекция 'lenin_works' найдена")
        except ValueError:
            self.collection = self.chroma_client.create_collection(
                name="lenin_works",
                embedding_function=embedding_function
            )
            logger.info("Создана новая коллекция 'lenin_works'")

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

            if not torch.cuda.is_available():
                logger.warning("CUDA недоступна! Генерация на CPU будет медленной")
                return "⚠️ Ошибка: GPU недоступна"


            query = f"{news_item.title} {news_item.content[:300]}"  # Уменьшаем длину запроса
            context = self.retrieve_context(query)

            prompt = f"""
                <s>[INST]Ты — Владимир Ильич Ленин. Проанализируй новость с позиции классовой борьбы и марксистской теории.
                Будь краток (3-5 предложений), используй революционную риторику. Выдели:
                - Классовые противоречия
                - Роль пролетариата
                - Исторические параллели с революционной борьбой

                Новость:
                {html.escape(news_item.title)}
                {html.escape(news_item.content[:800])}

                Контекст из работ Ленина:
                {context}[/INST]
                """

            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

            # Генерация с оптимизацией памяти
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.25,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

            full_response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Очистка памяти сразу после генерации
            del inputs
            torch.cuda.empty_cache()
            gc.collect()

            analysis_start = full_response.find("Анализ:")
            if analysis_start == -1:
                return full_response

            return full_response[analysis_start + len("Анализ:"):].strip()
        except torch.cuda.OutOfMemoryError:
            logger.error("Недостаточно памяти GPU для генерации анализа")
            torch.cuda.empty_cache()
            gc.collect()
            return "⚠️ Не удалось сгенерировать анализ: недостаточно памяти GPU"
        except Exception as e:
            logger.error(f"Ошибка генерации анализа: {str(e)}")
            return "Не удалось сгенерировать анализ."