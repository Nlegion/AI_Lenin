import logging
import re
import aiohttp
from pathlib import Path
from src.core.text_cleaner import TextCleaner
from src.core.analysis.context_orchestrator import AnalysisContextOrchestrator
from src.core.settings.analysis_defaults import (
    ANALYSIS_CACHE_LIMIT,
    LLAMA_SERVER_URL,
    default_generation_params,
)
from src.core.settings.config import Settings
from src.core.rag_system import get_rag_system
from typing import List
from src.core.retrieval.provider_factory import build_provider

logger = logging.getLogger(__name__)


class LeninAnalyzer:
    def __init__(self, vector_db_path: str = None):
        logger.info("Инициализация EnhancedLeninAnalyzer")
        _ = vector_db_path  # compatibility with legacy initializer signature
        self.config = Settings()
        self.server_url = LLAMA_SERVER_URL
        self.session = None
        self.rag_system = get_rag_system()
        self.analysis_cache = {}
        self.text_cleaner = TextCleaner()
        self.retrieval_provider = self._init_retrieval_provider()
        self.context_orchestrator = AnalysisContextOrchestrator(
            retrieval_provider=self.retrieval_provider,
            rag_system=self.rag_system,
        )

    def _init_retrieval_provider(self):
        config_path = Path(self.config.BASE_DIR) / "config" / "retrieval_pipeline.yaml"
        try:
            provider = build_provider(
                config_path=config_path,
                base_dir=Path(self.config.BASE_DIR),
                rag_system=self.rag_system,
            )
            if provider is None:
                logger.info("Retrieval provider disabled or unavailable. Using legacy RAG fallback.")
                return None
            logger.info("Retrieval provider initialized from pipeline config.")
            return provider
        except Exception as error:  # noqa: BLE001
            logger.exception("Failed to initialize retrieval provider: %s", error)
            return None

    async def initialize_session(self):
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=300, sock_connect=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    def extract_key_concepts(self, text: str) -> List[str]:
        """Извлечение ключевых концепций с акцентом на международную политэкономию"""
        concepts = []

        # Расширенный список терминов международной политэкономии
        political_economy_terms = [
            'капитал', 'пролетариат', 'буржуазия', 'эксплуатация',
            'революция', 'диалектика', 'материализм', 'идеализм',
            'классовая борьба', 'прибавочная стоимость', 'средства производства',
            'империализм', 'монополия', 'государство', 'диктатура пролетариата',
            'международная торговля', 'валютный кризис', 'рынок', 'капитализм',
            'социализм', 'коммунизм', 'колониализм', 'неоколониализм',
            'глобализация', 'национальный вопрос', 'санкции', 'экономические санкции',
            'международные отношения', 'дипломатия', 'гегемония', 'мировой рынок',
            'транснациональные корпорации', 'международное разделение труда',
            'внешняя политика', 'экономическая зависимость', 'сырьевая экономика',
            'финансовый капитал', 'долговая зависимость', 'неравномерное развитие'
        ]

        text_lower = text.lower()
        for term in political_economy_terms:
            if term in text_lower:
                concepts.append(term)

        # Дополнительные концепции based on content
        if any(word in text_lower for word in ['экономик', 'финанс', 'деньг', 'рынок', 'банк', 'валюта']):
            concepts.extend(['экономика', 'капитал', 'прибыль', 'политэкономия'])

        if any(word in text_lower for word in ['политик', 'власт', 'правительств', 'государств', 'партия']):
            concepts.extend(['политика', 'государство', 'власть', 'диктатура пролетариата'])

        if any(word in text_lower for word in ['международн', 'дипломати', 'санкц', 'договор', 'ООН', 'НАТО']):
            concepts.extend(['империализм', 'международные отношения', 'колониализм', 'международная политэкономия'])

        if any(word in text_lower for word in ['войн', 'военн', 'конфликт', 'оруж']):
            concepts.extend(['империализм', 'война', 'мирное сосуществование', 'военно-промышленный комплекс'])

        return list(set(concepts))[:5]

    async def generate_analysis(self, news_title: str, news_content: str, feedback: List[str] = None) -> str:
        try:
            await self.initialize_session()

            # Кэширование результатов (не кэшируем при наличии замечаний)
            cache_key = f"{news_title}_{hash(news_content[:200])}"
            if cache_key in self.analysis_cache and not feedback:
                return self.analysis_cache[cache_key]

            # Извлечение ключевых концепций
            key_concepts = self.extract_key_concepts(news_content)
            enhanced_query = f"{news_title} {news_content[:200]} {' '.join(key_concepts)}"

            # Многоуровневый поиск контекста (только если RAG система доступна)
            context = self.context_orchestrator.build_context(enhanced_query=enhanced_query)

            # Оптимизированный промпт с учетом замечаний
            system_prompt = self._create_optimized_prompt(context, news_title, news_content, feedback)
            user_content = f"Новость: {news_title}\n{news_content[:400]}"

            prompt = self._format_llama3_prompt(system_prompt, user_content)

            data = {"prompt": prompt, **default_generation_params()}

            async with self.session.post(
                    f"{self.server_url}/completion",
                    json=data,
                    headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get('content', '').strip()
                    cleaned_content = self.clean_analysis(content)

                    # Кэшируем результат только если нет замечаний
                    if not feedback:
                        self.analysis_cache[cache_key] = cleaned_content
                        if len(self.analysis_cache) > ANALYSIS_CACHE_LIMIT:
                            self.analysis_cache.clear()

                    return cleaned_content
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка сервера: {response.status} - {error_text}")
                    return "Анализ временно недоступен."

        except Exception as e:
            logger.exception(f"Ошибка генерации: {str(e)}")
            return "Ошибка анализа."

    def _create_optimized_prompt(self, context: str, news_title: str, news_content: str,
                                 feedback: List[str] = None) -> str:
        """Создание оптимизированного промпта с указанием версии"""
        base_prompt = (
            f"Ты — Владимир Ильич Ленин в 1923 году. Ты анализируешь современные события с позиции диалектического материализма и политэкономии.\n\n"
            f"Релевантные цитаты из моих работ для контекста:\n"
            f"{context}\n\n"
            "Строгие инструкции:\n"
            "1. Анализируй новости, связанные с экономикой, политикой, классовыми противоречиями, международными отношениями\n"
            "2. Особое внимание удели вопросам политики, империализма, колониализма, классовой борьбы\n"
            "3. Если новость касается только спорта, развлечений или культуры без классового подтекста - откажись от анализа\n"
            "4. Анализ должен быть кратким (3-4 предложения), конкретным и без общих фраз\n"
            "5. Избегай шаблонных вступлений\n"
            "6. Сфокусируйся на природе события, экономических противоречиях и империалистической практике\n"
            "7. Если в контексте есть релевантные цитаты - используй их для подкрепления анализа\n"
            "8. Формат цитирования: 'Как я писал в работе \"Название работы\": \"цитата\"'\n"
            "9. Будь аутентичным - используй характерные для Ленина термины и стиль\n"
            "10. ЗАВЕРШАЙ ВСЕ ПРЕДЛОЖЕНИЯ ПОЛНОСТЬЮ, БЕЗ ОБРЫВОВ\n"
            "11. Следи за грамотностью и избегай опечаток\n"
            "12. Всегда заканчивай анализ законченной мыслью\n"
            "13. Анализируй только свежие новости (не старше 24 часов)\n"
            "14. Особое внимание уделяй международной политэкономии: империализму, колониализму, международным экономическим отношениям\n"
            "15. ОБЯЗАТЕЛЬНО используй марксистско-ленинскую терминологию\n"
            "16. Анализ должен содержать не менее 3 предложений и быть законченным по смыслу\n"
            "17. НЕ используй аббревиатуры и сокращения.\n"
            "18. Если анализ невозможен, ответь строго: 'Данная тема не входит в круг моих исследований.'\n\n"

        )

        # Добавляем замечания из предыдущих попыток
        if feedback:
            feedback_text = "Учти следующие замечания из предыдущей попытки анализа:\n"
            for i, reason in enumerate(feedback, 1):
                feedback_text += f"{i}. {reason}\n"
            base_prompt += feedback_text + "\n"

        base_prompt += (
            "Пример хорошего анализа:\n"
            "'Экономические санкции против суверенных наций являются инструментом империалистического давления, характерным для высшей стадии капитализма. Как я писал в работе \"Империализм, как высшая стадия капитализма\": \"Вывоз капитала за границу, в отличие от вывоза товаров, приобретает совершенно исключительное значение\". Это отражает стремление финансового капитала к установлению гегемонии и контролю над ресурсами других наций. Модель Ай_Ленин v1.2.3'\n\n"
            "Формат ответа:\n"
            "- Если анализ возможен: сразу переходи к сути, начиная с характерного для Ленина резкого утверждения\n"
            "- Если анализ невозможен: 'Данная тема не входит в круг моих исследований.'\n\n"
            f"Новость: {news_title}\n{news_content[:400]}"
        )

        return base_prompt

    def clean_analysis(self, text: str) -> str:
        """Улучшенная очистка текста с обработкой обрывов"""
        if not text or text == "Анализ временно недоступен." or text == "Ошибка анализа.":
            return "Не удалось сгенерировать анализ."

        # Используем TextCleaner для исправления ошибок
        text = self.text_cleaner.clean_text(text)

        # Удаляем шаблонные фразы
        patterns = [
            r'Анализ новости с марксистско-ленинской точки зрения[:]?',
            r'Теперь[^.!?]*[.!?]', r'Рассмотрим[^.!?]*[.!?]',
            r'Анализируя[^.!?]*[.!?]', r'можно сделать вывод[^.!?]*[.!?]',
            r'данная ситуация[^.!?]*[.!?]', r'В контексте[^.!?]*[.!?]'
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Обрезаем до последнего законченного предложения
        text = self.text_cleaner.truncate_to_last_complete_sentence(text)

        # Дополнительная очистка
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]

        if not sentences:
            return "Не удалось сгенерировать анализ."

        # Собираем текст обратно
        result = '. '.join(sentences)

        # Убедимся, что текст заканчивается точкой
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
