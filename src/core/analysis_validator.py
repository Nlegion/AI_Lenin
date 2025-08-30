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
            r'в контексте новости[^.!?]*[.!?]',
            r'данная новость[^.!?]*[.!?]',
            r'с точки зрения[^.!?]*[.!?]',
            r'как мы видим[^.!?]*[.!?]'
        ]

        self.quote_bonus = 0.2

        # Запрещенные темы для анализа
        self.forbidden_topics = [
            "спорт", " теннис", "футбол", "хоккей", "развлечения",
            "знаменитости", "кино", "музыка", "искусство", "культура"
        ]

        # Ключевые марксистско-ленинские термины (расширенный список)
        self.marxist_terms = [
            'класс', 'капитал', 'пролетариат', 'буржуазия',
            'эксплуатация', 'революция', 'диалектика', 'материализм',
            'прибавочная стоимость', 'средства производства',
            'империализм', 'колониализм', 'неоколониализм',
            'международная торговля', 'финансовый капитал',
            'государство', 'власть', 'политика', 'экономика',
            'рынок', 'санкции', 'дипломатия', 'международные отношения',
            'рабочий', 'труд', 'зарплата', 'профсоюз', 'забастовка',
            'монополия', 'корпорация', 'кризис', 'инфляция', 'безработица',
            'социализм', 'коммунизм', 'капитализм', 'марксизм', 'ленинизм'
        ]

        # Минимальные требования к качеству
        self.min_length = 30
        self.min_sentences = 2
        self.min_marxist_terms = 0  # Убрали минимальное требование

        # Частые грамматические ошибки
        self.common_errors = [
            'капиталистическихого', 'буржуази', 'класовые',
            'класа', 'эксплуатаци', 'расмотрена', 'роси',
            'санкци', 'подчеркаивает', 'капиталистическ',
            'пролетариата', 'капиталистичес', 'иноваци',
            'иноваций', 'Франци', 'Парти', 'регистраци',
            'авиаци', 'современого', 'касационую'
        ]

    def validate_analysis(self, analysis: str, news_title: str = "") -> Dict:
        """Проверяет качество анализа и возвращает результат валидации"""
        validation_result = {
            "is_valid": False,
            "reasons": [],
            "score": 0.7,
            "has_quotes": False
        }

        refusal_phrases = [
            "не входит в круг моих исследований",
            "данная тема не подлежит анализу",
            "отказываюсь от анализа",
            "не подходит под задачу"
        ]

        if any(phrase in analysis.lower() for phrase in refusal_phrases):
            validation_result["reasons"].append("Модель отказалась от анализа")
            return validation_result

        # Проверка длины
        if len(analysis.strip()) < self.min_length:
            validation_result["reasons"].append(f"Слишком короткий анализ ({len(analysis)} символов)")
            return validation_result

        # Проверка количества предложений
        sentences = re.split(r'[.!?]+', analysis)
        valid_sentences = [s for s in sentences if len(s.strip()) > 10]
        if len(valid_sentences) < self.min_sentences:
            validation_result["reasons"].append("Недостаточно законченных предложений")

        # Проверка на запрещенные шаблоны
        for pattern in self.forbidden_patterns:
            if re.search(pattern, analysis, re.IGNORECASE):
                validation_result["reasons"].append("Обнаружены шаблонные фразы")
                break

        # Проверка на запрещенные темы
        analysis_lower = analysis.lower()
        for topic in self.forbidden_topics:
            if topic in analysis_lower:
                validation_result["reasons"].append(f"Обнаружена запрещенная тема: '{topic}'")
                break

        # СМЯГЧЕННАЯ проверка релевантной терминологии
        marxist_terms_count = sum(1 for term in self.marxist_terms if term in analysis.lower())

        # Только предупреждение, но не блокировка
        if marxist_terms_count == 0:
            validation_result["reasons"].append("Отсутствует марксистско-ленинская терминология")
        elif marxist_terms_count < 2:
            # Небольшой штраф к оценке, но не блокировка
            validation_result["score"] -= 0.1

        # Проверка на грамматические ошибки (только предупреждение, не блокировка)
        found_errors = []
        for error in self.common_errors:
            if error in analysis_lower:
                found_errors.append(error)

        if found_errors:
            validation_result["reasons"].append(f"Обнаружены грамматические ошибки: {', '.join(found_errors[:3])}")

        # Проверка связности с темой новости
        if news_title and not self._check_relevance(analysis, news_title):
            validation_result["reasons"].append("Низкая релевантность теме новости")

        # Проверка наличия цитат
        if 'как я писал' in analysis.lower() or 'в работе "' in analysis.lower():
            validation_result["has_quotes"] = True

        # Расчет общего скора - теперь менее строгий
        if len(validation_result["reasons"]) <= 3:  # Увеличили допустимое количество причин
            validation_result["is_valid"] = True
            validation_result["score"] = self._calculate_score(analysis, marxist_terms_count,
                                                               validation_result["has_quotes"])

            # Добавляем бонус за цитаты
            if validation_result["has_quotes"]:
                validation_result["score"] += self.quote_bonus

            validation_result["score"] = max(0.3, min(1.0, validation_result["score"]))
        else:
            validation_result["score"] = 0

        return validation_result

    def _check_relevance(self, analysis: str, news_title: str) -> bool:
        """Проверяет релевантность анализа теме новости"""
        # Извлекаем ключевые слова из заголовка
        title_keywords = set(re.findall(r'\w{4,}', news_title.lower()))
        analysis_keywords = set(re.findall(r'\w{4,}', analysis.lower()))

        # Ищем пересечение
        intersection = title_keywords.intersection(analysis_keywords)
        return len(intersection) >= 1  # Уменьшили требование до 1 совпадения

    def _calculate_score(self, analysis: str, marxist_terms_count: int, has_quotes: bool) -> float:
        """Рассчитывает оценку качества анализа"""
        length_score = min(1.0, len(analysis) / 200)  # Нормализуем длину
        terms_score = min(1.0, marxist_terms_count / 3)  # Нормализуем количество терминов

        # Считаем предложения
        sentences = re.split(r'[.!?]+', analysis)
        sentence_score = min(1.0, len(sentences) / 4)

        # Бонус за цитаты
        quote_score = 0.2 if has_quotes else 0

        # Средневзвешенная оценка
        return 0.4 * length_score + 0.3 * terms_score + 0.2 * sentence_score + quote_score