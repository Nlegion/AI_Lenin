import logging
import re
import aiohttp
import torch
import numpy as np
from functools import lru_cache
from src.core.settings.config import Settings
from src.core.rag_system import get_rag_system
from typing import List, Dict

logger = logging.getLogger(__name__)


class EnhancedLeninAnalyzer:
    def __init__(self, vector_db_path: str = None):
        logger.info("Инициализация EnhancedLeninAnalyzer")
        self.config = Settings()
        self.server_url = "http://127.0.0.1:8080"
        self.session = None
        self.rag_system = get_rag_system()
        self.analysis_cache = {}

    async def initialize_session(self):
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=300, sock_connect=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    def extract_key_concepts(self, text: str) -> List[str]:
        """Извлечение ключевых концепций для улучшения поиска"""
        concepts = []
        # Марксистско-ленинские термины для приоритетного поиска
        marxist_terms = [
            'капитал', 'пролетариат', 'буржуазия', 'эксплуатация',
            'революция', 'диалектика', 'материализм', 'идеализм',
            'классовая борьба', 'прибавочная стоимость', 'средства производства'
        ]

        text_lower = text.lower()
        for term in marxist_terms:
            if term in text_lower:
                concepts.append(term)

        return concepts[:3]  # Ограничиваем количество концепций

    async def generate_analysis(self, news_title: str, news_content: str) -> str:
        try:
            await self.initialize_session()

            # Кэширование результатов
            cache_key = f"{news_title}_{hash(news_content[:200])}"
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]

            # Извлечение ключевых концепций
            key_concepts = self.extract_key_concepts(news_content)
            enhanced_query = f"{news_title} {news_content[:200]} {' '.join(key_concepts)}"

            # Многоуровневый поиск контекста
            context = self.rag_system.retrieve_relevant_context(
                enhanced_query,
                k=5,
                author_filter="Ленин"
            )

            # Если контекст от Ленина недостаточен, добавляем других авторов
            if len(context.split()) < 100:
                additional_context = self.rag_system.retrieve_relevant_context(
                    enhanced_query,
                    k=3,
                    author_filter="МарксЭнгельс"
                )
                if additional_context:
                    context += "\n\n" + additional_context

            # Оптимизированный промпт
            system_prompt = self._create_optimized_prompt(context)
            user_content = f"Новость: {news_title}\n{news_content[:400]}"

            prompt = self._format_llama3_prompt(system_prompt, user_content)

            data = {
                "prompt": prompt,
                "n_predict": 150,
                "temperature": 0.7,  # Более творческая генерация
                "top_p": 0.85,
                "repeat_penalty": 1.3,
                "stop": ["<|eot_id|>", "\n\n", "###", "Теперь", "Рассмотрим"]
            }

            async with self.session.post(
                    f"{self.server_url}/completion",
                    json=data,
                    headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get('content', '').strip()
                    cleaned_content = self.clean_analysis(content)

                    # Кэшируем результат
                    self.analysis_cache[cache_key] = cleaned_content
                    if len(self.analysis_cache) > 1000:
                        self.analysis_cache.clear()

                    return cleaned_content
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка сервера: {response.status} - {error_text}")
                    return "Анализ временно недоступен."

        except Exception as e:
            logger.exception(f"Ошибка генерации: {str(e)}")
            return "Ошибка анализа."

    def _create_optimized_prompt(self, context: str) -> str:
        """Создание оптимизированного промпта"""
        return (
            "Ты — Владимир Ильич Ленин. Проанализируй новость с марксистско-ленинской точки зрения.\n\n"
            "Контекст для анализа:\n"
            f"{context}\n\n"
            "Инструкции:\n"
            "1. Дайте краткий анализ (2-3 предложения)\n"
            "2. Сфокусируйтесь на классовых отношениях и экономических противоречиях\n"
            "3. Избегайте общих фраз и шаблонных выражений\n"
            "4. Будьте конкретны и аутентичны\n"
            "5. Начинайте сразу с сути анализа\n"
            "6. Используйте диалектический подход\n"
            "7. Укажите на основные противоречия\n\n"
            "Запрещенные фразы: 'теперь', 'рассмотрим', 'можно сделать вывод', "
            "'данная ситуация', 'в контексте новости'"
        )

    def clean_analysis(self, text: str) -> str:
        """Улучшенная очистка текста от шаблонных фраз"""
        if not text:
            return "Не удалось сгенерировать анализ."

        # Удаление шаблонных фраз
        patterns = [
            r'Теперь[^.!?]*[.!?]', r'Рассмотрим[^.!?]*[.!?]',
            r'Анализируя[^.!?]*[.!?]', r'можно сделать вывод[^.!?]*[.!?]',
            r'данная ситуация[^.!?]*[.!?]', r'В контексте[^.!?]*[.!?]',
            r'Как отмечал[^.!?]*[.!?]', r'С точки зрения[^.!?]*[.!?]'
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Удаление повторяющихся фраз
        text = re.sub(r'(.+?)(\1+)', r'\1', text)

        # Выбор лучших предложений
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        if len(sentences) > 3:
            sentences = sentences[:3]  # Ограничиваем тремя предложениями

        return '. '.join(sentences) + '.'

    def _format_llama3_prompt(self, system_prompt: str, user_input: str) -> str:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_input}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )