import torch
import chromadb
import hashlib
import joblib
import re
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from typing import List, Dict, Tuple
from peft import PeftModel


class LeninStoppingCriteria(StoppingCriteria):
    """Критерии остановки генерации в стиле Ленина"""

    def __init__(self, tokenizer, max_length=3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.key_phrases = [
            "товарищи",
            "пролетариат",
            "революция",
            "капитализм",
            "социализм"
        ]
        self.encoded_phrases = [tokenizer(phrase, add_special_tokens=False)['input_ids']
                                for phrase in self.key_phrases]

    def __call__(self, input_ids, scores, **kwargs):
        # Останавливаем генерацию при наличии ключевых фраз в конце
        last_tokens = input_ids[0][-self.max_length:]
        for phrase in self.encoded_phrases:
            if len(last_tokens) >= len(phrase) and \
                    last_tokens[-len(phrase):].tolist() == phrase:
                return True
        return False


class LeninAI:
    def __init__(self):
        # Конфигурация 4-битного квантования
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Инициализация модели эмбеддингов
        self.embed_model = SentenceTransformer(
            'intfloat/multilingual-e5-large',
            device='cuda'
        )

        # Инициализация векторной БД
        self.chroma_client = chromadb.PersistentClient(path="vector_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="lenin_works",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='intfloat/multilingual-e5-large'
            )
        )

        # Загрузка модели
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        # Критерии остановки
        self.stopping_criteria = LeninStoppingCriteria(self.tokenizer)

        # Кэширование
        self.cache = joblib.Memory(location="cache_dir", verbose=0)
        self.response_cache = {}
        self.stats = {"total_queries": 0, "cache_hits": 0}

    def retrieve_context(self, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """Поиск релевантных фрагментов с расширенной фильтрацией"""
        try:
            # Используем расширенный запрос
            expanded_query = f"{query} марксизм ленинизм теория"

            results = self.collection.query(
                query_texts=[expanded_query],
                n_results=k * 3,  # Берем больше результатов
                include=["documents", "metadatas"]
            )

            context_parts = []
            sources = []

            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                # Более мягкая проверка релевантности
                if self.is_relevant(doc, query) or self.is_authentic(doc):
                    source_info = {
                        "title": meta.get('source_title', 'ПСС Ленина'),
                        "volume": meta.get('volume', 'N/A'),
                        "pages": f"{meta.get('page_start', '?')}-{meta.get('page_end', '?')}"
                    }
                    context_parts.append(doc)  # Только текст, без метаданных
                    sources.append(source_info)

                    if len(context_parts) >= k:
                        break

            context_text = "\n\n".join(context_parts) if context_parts else ""
            return context_text, sources
        except Exception as e:
            print(f"Ошибка при поиске контекста: {str(e)}")
            return "", []

    # def is_relevant(self, text: str, query: str) -> bool:
    #     """Проверка релевантности через комбинацию методов"""
    #     # Быстрая проверка по ключевым словам
    #     query_keywords = set(query.lower().split())
    #     text_keywords = set(text.lower().split())
    #     common_keywords = query_keywords & text_keywords
    #
    #     if common_keywords:
    #         return True
    #
    #     # Семантическая проверка для сложных случаев
    #     query_embed = self.embed_model.encode([query])[0]
    #     text_embed = self.embed_model.encode([text])[0]
    #     similarity = torch.nn.functional.cosine_similarity(
    #         torch.tensor(query_embed).unsqueeze(0),
    #         torch.tensor(text_embed).unsqueeze(0)
    #     )
    #
    #     return similarity.item() > 0.5

    def is_relevant(self, text: str, query: str) -> bool:
        """Упрощенная проверка релевантности"""
        query_words = set(query.lower().split())
        text_lower = text.lower()
        return any(word in text_lower for word in query_words)

    def is_authentic(self, text: str) -> bool:
        """Фильтрация неаутентичных фрагментов"""
        # Исключаем редакторские примечания и комментарии
        if "примечание редакции" in text.lower():
            return False

        # Исключаем фрагменты без характерной ленинской лексики
        key_terms = ["пролетариат", "революция", "буржуазия", "марксизм", "диктатура"]
        return any(term in text.lower() for term in key_terms)

    def refine_lenin_style(self, text: str) -> str:
        """Приведение текста к характерному ленинскому стилю"""
        # Удаление современных и иностранных выражений
        replacements = {
            r'\bангл?\.': '',
            r'\betc\b': '',
            r'\bнапример\b': 'таким образом',
            r'\bочень\b': 'весьма',
            r'\bможно\b': 'следует',
            r'\bнужно\b': 'необходимо',
            r'\bпотому что\b': 'ибо',
            r'\bкак бы\b': '',
            r'\bтипа\b': ''
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Добавление характерных ленинских оборотов
        if not text.startswith(("Товарищи!", "Пролетарии!", "Вопрос")):
            text = "Товарищи! " + text

        # Упрощение сложных конструкций
        text = re.sub(r", которая", ", что", text)
        text = re.sub(r", который", ", что", text)

        # Обеспечение краткости (2-3 предложения)
        sentences = re.split(r'(?<=[.!?]) +', text)
        if len(sentences) > 3:
            text = ' '.join(sentences[:3])

        return text

    def validate_response(self, response: str, sources: List[Dict]) -> str:
        """Базовая проверка ответа"""
        if "не освещён" in response or "не упоминается" in response:
            return "В моих работах этот вопрос не освещён."
        return response

    # def validate_response(self, response: str, sources: List[Dict]) -> str:
    #     """Проверка ответа на соответствие источникам"""
    #     # Если нет источников, сразу возвращаем стандартный ответ
    #     if not sources:
    #         return "В моих работах этот вопрос не освещён."
    #
    #     # Проверяем, содержит ли ответ ключевые термины из контекста
    #     if "не освещён" in response or "не упоминается" in response:
    #         return response
    #
    #     # Собираем все уникальные термины из источников
    #     source_terms = set()
    #     for source in sources:
    #         if 'title' in source:
    #             # Берем только значимые слова (длиннее 4 символов)
    #             terms = [word for word in source['title'].lower().split() if len(word) > 4]
    #             source_terms.update(terms)
    #
    #     # Если нет терминов для проверки, возвращаем ответ как есть
    #     if not source_terms:
    #         return response
    #
    #     # Проверяем соответствие ответа источникам
    #     response_terms = set(response.lower().split())
    #     common_terms = source_terms & response_terms
    #
    #     if not common_terms:
    #         # Попробуем более мягкую проверку через подстроки
    #         for term in source_terms:
    #             if term in response.lower():
    #                 return response
    #
    #         # Если совсем не совпадает, возвращаем стандартный ответ
    #         return "В моих работах этот вопрос не освещён."
    #
    #     return response

    def generate_response(self, query: str) -> Tuple[str, List[Dict]]:
        """Генерация ответа в стиле Ленина с источниками"""
        self.stats["total_queries"] += 1

        # Проверка кэша
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self.response_cache:
            self.stats["cache_hits"] += 1
            return self.response_cache[query_hash]

        # Фильтр неуместных вопросов
        inappropriate_keywords = ["личная жизнь", "семья", "любовь", "смерть"]
        if any(kw in query.lower() for kw in inappropriate_keywords):
            return "Товарищ, эти вопросы не относятся к классовой борьбе!", []

        # Поиск контекста
        context, sources = self.retrieve_context(query)

        # Формирование промпта
        # prompt = f"""
        # <s>[INST] Ты — Владимир Ильич Ленин (1923 год). Отвечай как на партсобрании.
        # Отвечай кратко (2-3 предложения), используя марксистско-ленинскую терминологию.
        # Всегда опирайся на предоставленные источники.
        # Если вопрос не связан с политикой или философией, вежливо откажись отвечать.
        # Если информации недостаточно, скажи "В моих работах этот вопрос не освещён".
        #
        # Контекст:
        # {context}
        #
        # Вопрос: {query} [/INST]
        # Ответ:
        # """

        prompt = f"""
        <s>[INST] Ты — Владимир Ильич Ленин. Отвечай на вопросы кратко и точно, используя предоставленный контекст.
        Если информации в контексте недостаточно, скажи "В моих работах этот вопрос не освещён".

        Контекст:
        {context}

        Вопрос: {query} [/INST]
        Ответ:
        """

        # print(f"\n===== Сгенерированный промпт =====")
        # print(prompt)
        # print("=" * 50)

        # Генерация ответа
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            repetition_penalty=1.2,
            stopping_criteria=StoppingCriteriaList([self.stopping_criteria])
        )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлечение ответа
        response = full_response.split("Ответ:")[-1].strip()

        # Постобработка
        response = self.refine_lenin_style(response)
        response = self.validate_response(response, sources)

        # Кэширование
        self.response_cache[query_hash] = (response, sources)
        return response, sources

    def get_stats(self) -> dict:
        """Статистика производительности"""
        return {
            "total_queries": self.stats["total_queries"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": f"{(self.stats['cache_hits'] / self.stats['total_queries']) * 100:.1f}%"
            if self.stats["total_queries"] > 0 else "0%"
        }


if __name__ == "__main__":
    ai = LeninAI()
    print("Система инициализирована. Пример запроса:")
    response, sources = ai.generate_response("Объясните теорию империализма")
    print(f"Ответ: {response}")
    print(f"Источники: {sources}")