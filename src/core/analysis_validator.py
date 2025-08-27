import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class AnalysisValidator:
    def __init__(self):
        # Запрещенные фразы и шаблоны
        self.forbidden_patterns = [
            r'теперь[^.!?]*[.!?]',
            r'рассмотрим[^.!?]*[.!?]',
            r'анализируя[^.!?]*[.!?]',
            r'можно сделать вывод[^.!?]*[.!?]',
            r'данная ситуация[^.!?]*[.!?]',
            r'в контексте новости[^.!?]*[.!?]'
        ]
        self.forbidden_topics = [
            "спорт", "теннис", "футбол", "хоккей", "развлечения",
            "знаменитости", "кино", "музыка", "искусство"
        ]

        # Ключевые марксистско-ленинские термины
        self.marxist_terms = [
            'класс', 'капитал', 'пролетариат', 'буржуазия',
            'эксплуатация', 'революция', 'диалектика', 'материализм',
            'прибавочная стоимость', 'средства производства'
        ]

        # Минимальные требования к качеству
        self.min_length = 50
        self.min_sentences = 2
        self.min_marxist_terms = 1

    def validate_analysis(self, analysis: str, news_title: str = "") -> Dict:
        """Проверяет качество анализа и возвращает результат валидации"""
        validation_result = {
            "is_valid": False,
            "reasons": [],
            "score": 0
        }

        # Проверка длины
        if len(analysis.strip()) < self.min_length:
            validation_result["reasons"].append(f"Слишком короткий анализ ({len(analysis)} символов)")
            return validation_result

        # Проверка количества предложений
        sentences = re.split(r'[.!?]+', analysis)
        if len([s for s in sentences if len(s.strip()) > 10]) < self.min_sentences:
            validation_result["reasons"].append("Недостаточно законченных предложений")

        # Проверка на запрещенные шаблоны
        for pattern in self.forbidden_patterns:
            if re.search(pattern, analysis, re.IGNORECASE):
                validation_result["reasons"].append("Обнаружены шаблонные фразы")
                break

        # Проверка релевантной терминологии
        marxist_terms_count = sum(1 for term in self.marxist_terms if term in analysis.lower())
        if marxist_terms_count < self.min_marxist_terms:
            validation_result["reasons"].append("Недостаточно марксистско-ленинской терминологии")

        # Проверка связности с темой новости
        if news_title and not self._check_relevance(analysis, news_title):
            validation_result["reasons"].append("Низкая релевантность теме новости")

        # Расчет общего скора
        if not validation_result["reasons"]:
            validation_result["is_valid"] = True
            validation_result["score"] = self._calculate_score(analysis, marxist_terms_count)
        else:
            validation_result["score"] = 0

        return validation_result

    def _check_relevance(self, analysis: str, news_title: str) -> bool:

        # Проверка на запрещенные темы
        analysis_lower = analysis.lower()
        for topic in self.forbidden_topics:
            if topic in analysis_lower:
                return False

        """Проверяет релевантность анализа теме новости"""
        # Извлекаем ключевые слова из заголовка
        title_keywords = set(re.findall(r'\w{4,}', news_title.lower()))
        analysis_keywords = set(re.findall(r'\w{4,}', analysis.lower()))

        # Ищем пересечение
        intersection = title_keywords.intersection(analysis_keywords)
        return len(intersection) >= 2  # Минимум 2 общих значимых слова

    def _calculate_score(self, analysis: str, marxist_terms_count: int) -> float:
        """Рассчитывает оценку качества анализа"""
        length_score = min(1.0, len(analysis) / 200)  # Нормализуем длину
        terms_score = min(1.0, marxist_terms_count / 3)  # Нормализуем количество терминов

        # Считаем предложения
        sentences = re.split(r'[.!?]+', analysis)
        sentence_score = min(1.0, len(sentences) / 4)

        # Средневзвешенная оценка
        return 0.4 * length_score + 0.4 * terms_score + 0.2 * sentence_score