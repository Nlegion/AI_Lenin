import logging
import re
import time
import os
import gc
import torch
import chromadb
import numpy as np
from llama_cpp import Llama
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb import PersistentClient
from functools import lru_cache
import platform

# Настройка абсолютных путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Корень проекта
UNIT_TEST_DIR = os.path.join(BASE_DIR, "unit_testing")  # Директория тестов

# Создаем директорию для логов, если ее нет
os.makedirs(UNIT_TEST_DIR, exist_ok=True)

# Настройка логирования с абсолютными путями
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(UNIT_TEST_DIR, "model_test.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LeninAnalyzer")


class LeninAnalyzer:
    def __init__(self, vector_db_path: str = None):
        logger.info("Инициализация LeninAnalyzer")

        # Конфигурация с абсолютными путями
        self.config = {
            "model_path": os.path.join(BASE_DIR, "models", "saiga", "model-q4_K.gguf"),
            "lora_path": os.path.join(BASE_DIR, "models", "saiga", "lora_adapter.gguf"),
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

            # Определение оптимальных параметров
            n_gpu_layers = 0
            n_threads = 4

            # Для систем с достаточной VRAM
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_vram > 10:
                    n_gpu_layers = 20
                    logger.warning("Экспериментальный режим: использование GPU слоев")

            self.llm = Llama(
                model_path=self.config["model_path"],
                lora_path=self.config["lora_path"] if os.path.exists(self.config["lora_path"]) else None,
                lora_base=None,
                n_ctx=4096,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format="llama-3",
                verbose=False,
                n_batch=128,
                use_mlock=True
            )
            logger.info(f"Модель успешно загружена (GPU слои: {n_gpu_layers}, потоки: {n_threads})")
        except Exception as e:
            logger.exception(f"Ошибка загрузки модели: {str(e)}")

            # Попытка загрузки в безопасном режиме
            logger.warning("Попытка загрузки в безопасном режиме (CPU only)...")
            self.llm = Llama(
                model_path=self.config["model_path"],
                n_ctx=2048,
                n_gpu_layers=0,
                n_threads=1,
                chat_format="llama-3",
                verbose=False,
                n_batch=1
            )
            logger.info("Модель загружена в безопасном режиме")

    def unload_model(self):
        if self.llm is not None:
            logger.info("Выгрузка модели для освобождения памяти")
            del self.llm
            self.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Очищена память CUDA")

    def _init_vector_db(self, path: str):
        # Инициализируем функцию эмбеддингов
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.config["embedding_model"],
            device="cpu"
        )

        self.chroma_client = PersistentClient(path=path)
        try:
            self.collection = self.chroma_client.get_collection(
                name="lenin_works",
                embedding_function=self.embedding_function
            )
            logger.info("Коллекция lenin_works успешно загружена")

            # Проверка количества элементов
            count = self.collection.count()
            logger.info(f"В коллекции {count} документов")
        except Exception as e:
            logger.error(f"Ошибка загрузки коллекции: {str(e)}")
            self.collection = None

    @lru_cache(maxsize=500)
    def cached_embedding(self, text: str) -> list:
        if self.embedding_function is None:
            return []
        return self.embedding_function([text])[0]

    def retrieve_context(self, query: str, k: int = 3) -> str:  # Увеличили количество контекстов
        try:
            if self.collection is None:
                logger.warning("Векторная БД не инициализирована")
                return ""

            count = self.collection.count()
            if count == 0:
                logger.warning("Векторная БД пуста")
                return ""

            embedding = self.cached_embedding(query)

            # Исправленная проверка эмбеддинга
            if embedding is None or (isinstance(embedding, np.ndarray) and embedding.size == 0):
                logger.error("Не удалось получить эмбеддинг для запроса")
                return ""

            # Проверка типа эмбеддинга
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["documents"]
            )

            # Обработка результатов ChromaDB
            context_parts = []
            if 'documents' in results and results['documents'] is not None:
                documents = results['documents']
                if isinstance(documents, list) and len(documents) > 0:
                    for doc_list in documents:
                        if isinstance(doc_list, list):
                            for doc in doc_list:
                                if doc and isinstance(doc, str):
                                    context_parts.append(doc)

            context = " ".join(context_parts[:500])  # Ограничиваем общую длину
            if context:
                logger.debug(f"Контекст: {context[:100]}...")
                return context

            return ""
        except Exception as e:
            logger.exception(f"Ошибка поиска контекста: {str(e)}")
            return ""

    def clean_analysis(self, text: str) -> str:
        # Удаление артефактов генерации
        text = re.sub(
            r'(\[INST\].*?\[/INST\]|<\/?s>|Анализ Ленина:|```|SYSTEM:|USER:|ASSISTANT:|\([^)]*\)|"|«|»)',
            '',
            text
        )
        text = re.sub(r'Владимир Ильич (Ленин)?[:,]?\s*', '', text, flags=re.IGNORECASE)

        # Обеспечение ровно 2 предложений
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        cleaned = ' '.join(sentences[:2]).strip()

        # Фильтрация пустых предложений
        if not cleaned or len(cleaned.split()) < 3:
            return "Анализ не удался. Попробуйте другую новость."
        return cleaned

    def generate_analysis(self, news_title: str, news_content: str) -> str:
        try:
            if self.llm is None:
                self.reload_model()

            # Получение контекста
            query = f"{news_title} {news_content[:100]}"
            context = self.retrieve_context(query)

            # Формируем промпт
            system_prompt = (
                "Ты — Владимир Ильич Ленин. Дай марксистский анализ классовой борьбы ровно в 2 предложениях. "
                "Используй революционный стиль без упоминания стран и национальностей. "
                "В ответе используй термины: пролетариат, буржуазия, революция, эксплуатация."
            )

            if context:
                user_content = f"Контекст: {context}\n\nНовость: {news_title}\n{news_content[:500]}"
            else:
                user_content = f"Новость: {news_title}\n{news_content[:500]}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            logger.debug(f"Промпт: {str(messages)[:300]}...")

            # Генерация ответа
            start_time = time.time()
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                top_p=0.85,
                max_tokens=150,
                stop=["<|eot_id|>", "\n\n"],
                repeat_penalty=1.1
            )
            gen_time = time.time() - start_time

            # Извлечение и очистка ответа
            analysis = response['choices'][0]['message']['content'].strip()
            cleaned = self.clean_analysis(analysis)

            logger.info(f"Генерация заняла {gen_time:.2f} секунд")
            return cleaned

        except Exception as e:
            logger.exception(f"Ошибка генерации: {str(e)}")
            return "Не удалось сгенерировать анализ."


def run_test_suite():
    """Запускает набор тестов для проверки работы системы"""
    logger.info("Запуск тестовой среды")

    # Используем абсолютный путь к векторной БД
    analyzer = LeninAnalyzer(
        vector_db_path=os.path.join(BASE_DIR, "database", "vector_db")
    )

    # Тестовые случаи с революционным контекстом
    test_cases = [
        {
            "title": "Забастовка рабочих на заводе",
            "content": "Рабочие автомобильного завода остановили производство, требуя повышения зарплат и улучшения условий труда.",
            "expected_keywords": ["пролетариат", "эксплуатация", "борьба", "революция"]
        },
        {
            "title": "Капиталистическая эксплуатация",
            "content": "Крупные корпорации увеличили рабочий день до 12 часов без повышения оплаты труда.",
            "expected_keywords": ["буржуазия", "эксплуатация", "революция", "класс"]
        },
        {
            "title": "Мировой экономический кризис",
            "content": "Банки объявили о массовых увольнениях из-за экономического спада и падения спроса.",
            "expected_keywords": ["капитализм", "кризис", "пролетариат", "революция"]
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'=' * 50}\nТест #{i}: {test_case['title']}\n{'=' * 50}")
        start_time = time.time()

        try:
            analysis = analyzer.generate_analysis(test_case["title"], test_case["content"])
            elapsed = time.time() - start_time

            # Проверка формата
            sentences = re.split(r'(?<=[.!?])\s+', analysis)
            sentence_count = len(sentences)

            # Проверка ключевых слов
            found_keywords = [
                kw for kw in test_case["expected_keywords"]
                if kw.lower() in analysis.lower()
            ]

            # Вывод результатов
            print(f"\n[Тест #{i}] {test_case['title']} ({elapsed:.2f} сек)")
            print(
                f"Предложения: {sentence_count} | Ключевые слова: {len(found_keywords)}/{len(test_case['expected_keywords'])}")
            print("-" * 50)
            print(analysis)
            print("=" * 50)

            logger.info(f"Результат анализа ({elapsed:.2f} сек):")
            logger.info(f"Кол-во предложений: {sentence_count}")
            logger.info(f"Найдены ключевые слова: {', '.join(found_keywords) if found_keywords else 'НЕТ'}")
            logger.info(f"Анализ:\n{analysis}")

        except Exception as e:
            logger.error(f"Ошибка при выполнении теста #{i}: {str(e)}")

    analyzer.unload_model()
    logger.info("Тестирование завершено")


if __name__ == "__main__":
    logger.info(f"Запуск тестовой системы анализа | Рабочая директория: {BASE_DIR}")
    logger.info(f"Логи сохраняются в: {os.path.join(UNIT_TEST_DIR, 'model_test.log')}")

    print(f"Проверка окружения:")
    print(f"ОС: {platform.system()} {platform.release()}")
    print(f"Архитектура: {platform.machine()}")
    print(f"Процессор: {platform.processor()}")
    print(f"PyTorch CUDA доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Устройство CUDA: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print(f"Chromadb версия: {chromadb.__version__}")

    run_test_suite()
    logger.info("Все тесты завершены")