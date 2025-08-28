import logging
import re
import aiohttp
import torch
import numpy as np
from src.core.text_cleaner import TextCleaner
from src.core.settings.config import Settings
from src.core.rag_system import get_rag_system
from typing import List, Dict

logger = logging.getLogger(__name__)


class LeninAnalyzer:
    def __init__(self, vector_db_path: str = None):
        logger.info("Инициализация EnhancedLeninAnalyzer")
        self.config = Settings()
        self.server_url = "http://127.0.0.1:8080"
        self.session = None
        self.rag_system = get_rag_system()
        self.analysis_cache = {}
        self.text_cleaner = TextCleaner()

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

            # Многоуровневый поиск контекста (только если RAG система доступна)
            context = ""
            if self.rag_system is not None:
                try:
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
                except Exception as e:
                    logger.error(f"Ошибка RAG поиска: {str(e)}")
                    # Продолжаем без контекста, если RAG система недоступна
                    context = ""

            # Оптимизированный промпт
            system_prompt = self._create_optimized_prompt(context, news_title, news_content)
            user_content = f"Новость: {news_title}\n{news_content[:400]}"

            prompt = self._format_llama3_prompt(system_prompt, user_content)

            data = {
                "prompt": prompt,
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 40,
                "repeat_penalty": 1.5,
                "typical_p": 0.9,
                "stop": ["<|eot_id|>", "\n\n", "###", "Теперь", "Рассмотрим", "Анализируя"],
                "n_predict": 150,
                "mirostat": 2,
                "mirostat_tau": 3.0,
                "mirostat_eta": 0.1
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

    def _create_optimized_prompt(self, context: str, news_title: str, news_content: str) -> str:
        """Создание оптимизированного промпта с акцентом на цитирование Ленина"""
        return (
            "Ты — Владимир Ильич Ленин в 1923 году. Ты анализируешь современные события с позиции диалектического материализма.\n\n"
            "Контекст для анализа (цитаты из моих произведений):\n"
            f"{context}\n\n"
            "Строгие инструкции:\n"
            "1. Анализируй новости, связанные с экономикой, политикой, классовыми противоречиями, международными отношениями и социальными вопросами\n"
            "2. Если новость касается только спорта, развлечений или культуры без классового подтекста - откажись от анализа\n"
            "3. Анализ должен быть кратким (2-3 предложения), конкретным и аутентичным\n"
            "4. ОБЯЗАТЕЛЬНО используй прямые цитаты из моих работ, когда это уместно\n"
            "5. Цитирование должно быть органичным и соответствовать контексту новости\n"
            "6. Указывай источник цитаты в скобках (например: ПСС, 5 изд., т. 45, с. 82)\n"
            "7. Сфокусируйся на классовой природе события, экономических противоречиях и империалистической практике\n"
            "8. Будь аутентичным - используй характерные для моего стиля термины и выражения\n"
            "9. Завершай предложения полностью, без обрывов\n"
            "10. Следи за грамотностью\n\n"
            "Пример хорошего анализа с цитированием:\n"
            "'Увеличение прибылей капиталистов при росте эксплуатации рабочих - классическое проявление империалистической стадии капитализма. Как я отмечал в \"Империализме как высшей стадии капитализма\": \"Монополии, олигархия, стремление к господству вместо стремления к свободе... - вот что характеризует империализм\" (ПСС, 5 изд., т. 27, с. 387). Буржуазия наращивает накопление капитала через усиление давления на пролетариат.'\n\n"
            "Формат ответа:\n"
            "- Если анализ возможен: сразу переходи к сути, используя цитаты где уместно\n"
            "- Если анализ невозможен: 'Данная тема не входит в круг моих исследований.'\n\n"
            f"Новость: {news_title}\n{news_content[:400]}"
        )

    def clean_analysis(self, text: str) -> str:
        """Улучшенная очистка текста с обработкой обрывов"""
        if not text:
            return "Не удалось сгенерировать анализ."

        # Используем TextCleaner для исправления ошибок
        text = self.text_cleaner.clean_text(text)

        # Обработка обрывов слов
        text = re.sub(r'(\w+)\s*\.\s*$', r'\1.', text)  # Убираем пробелы перед точкой в конце
        text = re.sub(r'(\w+)\.\s*$', r'\1.', text)  # Исправляем обрывы слов с точкой

        # Удаляем явные обрывы предложений
        text = re.sub(r'\b(эксплу|капиталист|империалист|пролетар|буржуаз|революц|социалист|коммунист)\.',
                      r'\1и.', text)

        # Завершаем оборванные предложения
        if text.endswith(('и эксплу', 'и капиталист', 'и империалист')):
            text += "ация."
        elif text.endswith(('и пролетар', 'и буржуаз', 'и революц', 'и социалист', 'и коммунист')):
            text += "ия."

        # Выбор лучших предложений
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        if len(sentences) > 3:
            sentences = sentences[:3]

        result = '. '.join(sentences) + '.'

        # Убедимся, что результат заканчивается точкой
        if not result.endswith('.'):
            result += '.'

        return result

    def _format_llama3_prompt(self, system_prompt: str, user_input: str) -> str:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_input}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )