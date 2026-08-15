import re
import logging
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)


class NewsClassifier:
    def __init__(self):
        # Обновленные категории и ключевые слова
        self.categories = {
            "politics": [
                "политика",
                "правительство",
                "выборы",
                "партия",
                "власть",
                "государство",
                "президент",
                "министр",
                "кремль",
                "санкции",
            ],
            "economics": [
                "экономика",
                "финансы",
                "рынок",
                "бизнес",
                "компания",
                "капитал",
                "инвестиции",
                "банк",
                "валюта",
                "бюджет",
            ],
            "social": [
                "общество",
                "социальный",
                "труд",
                "работа",
                "зарплата",
                "пенсия",
                "медицина",
                "образование",
                "здравоохранение",
            ],
            "international": [
                "международный",
                "дипломатия",
                "ООН",
                "НАТО",
                "ЕС",
                "санкции",
                "посол",
                "договор",
                "переговоры",
            ],
            "culture": [
                "культура",
                "искусство",
                "кино",
                "театр",
                "литература",
                "музей",
                "выставка",
                "концерт",
                "фестиваль",
            ],
            "other": ["технология", "спорт", "развлечение", "наука"],
        }

        # Приоритетные категории для анализа
        self.high_priority_categories = [
            "politics",
            "economics",
            "social",
            "international",
        ]
        self.low_priority_categories = ["culture"]
        self.skip_categories = ["other"]

        # Инициализация ML модели
        self.model = self._train_model()

    def _train_model(self):
        """Улучшенное обучение модели с большим количеством примеров"""
        texts = []
        labels = []

        # Добавляем больше примеров для каждой категории
        for category, keywords in self.categories.items():
            # Базовые ключевые слова
            for keyword in keywords:
                texts.append(keyword)
                labels.append(category)

            # Комбинации ключевых слов
            for i in range(len(keywords)):
                for j in range(i + 1, min(i + 3, len(keywords))):
                    combined = f"{keywords[i]} {keywords[j]}"
                    texts.append(combined)
                    labels.append(category)

            # Добавляем контекстные примеры
            context_examples = {
                "politics": [
                    "заседание правительства",
                    "встреча с президентом",
                    "политическое решение",
                    "государственная программа",
                ],
                "economics": [
                    "финансовый отчет",
                    "экономический рост",
                    "рыночные показатели",
                    "бизнес проект",
                ],
                "social": [
                    "социальная защита",
                    "трудовые отношения",
                    "медицинское обслуживание",
                    "образовательная программа",
                ],
                "international": [
                    "международные отношения",
                    "дипломатическая миссия",
                    "встреча на высшем уровне",
                    "международный договор",
                ],
                "culture": [
                    "культурное мероприятие",
                    "художественная выставка",
                    "театральная постановка",
                    "литературный вечер",
                ],
            }

            if category in context_examples:
                for example in context_examples[category]:
                    texts.append(example)
                    labels.append(category)

        # Создаем и обучаем модель
        model = make_pipeline(
            TfidfVectorizer(max_features=1000, ngram_range=(1, 2)),
            MultinomialNB(alpha=0.1),
        )
        model.fit(texts, labels)

        return model

    def classify_news(self, title: str, content: str) -> Tuple[str, float]:
        """Улучшенная классификация новости"""
        # Объединяем заголовок и содержание
        text = f"{title} {content}".lower()

        # Удаляем стоп-слова и короткие слова
        words = re.findall(r"\b\w{4,}\b", text)
        processed_text = " ".join(words)

        # Предсказание категории
        try:
            category = self.model.predict([processed_text])[0]
            probability = max(self.model.predict_proba([processed_text])[0])
            return category, probability
        except Exception as error:
            logger.warning("news_classify_failed", error=str(error))
            return "other", 0.5

    def should_analyze(self, title: str, content: str) -> Tuple[bool, str]:
        """Определяет, нужно ли анализировать новость"""
        text = f"{title} {content}".lower()

        # Жесткий фильтр по ключевым словам - отсеиваем спорт, развлечения и т.д.
        exclude_keywords = [
            "теннис",
            "футбол",
            "хоккей",
            "спорт",
            "матч",
            "соревнование",
            "чемпионат",
            "актер",
            "певец",
            "кино",
            "фильм",
            "музыка",
            "концерт",
            "выставка",
            "искусство",
            "развлечения",
            "культура",
        ]

        if any(keyword in text for keyword in exclude_keywords):
            return False, "Пропуск: тема не подходит для анализа"

        # Проверяем наличие экономических и политических терминов
        economic_terms = ["экономик", "финанс", "бизнес", "рынок", "компани", "банк"]
        political_terms = [
            "политик",
            "правительств",
            "президент",
            "министр",
            "выбор",
            "партия",
        ]
        social_terms = ["общество", "социальн", "труд", "работа", "зарплата", "пенсия"]

        priority_terms = economic_terms + political_terms + social_terms

        # Если есть хотя бы один приоритетный термин, анализируем
        if any(term in text for term in priority_terms):
            return True, "Приоритетная тема для анализа"

        # Для остальных новостей используем ML классификатор
        category, confidence = self.classify_news(title, content)

        if category in self.skip_categories:
            return (
                False,
                f"Пропуск: категория '{category}' (уверенность: {confidence:.2f})",
            )

        if category in self.low_priority_categories and confidence < 0.6:
            return (
                False,
                f"Пропуск: низкая уверенность в категории '{category}' (уверенность: {confidence:.2f})",
            )

        if confidence < 0.4:
            return (
                False,
                f"Пропуск: общая низкая уверенность классификации (уверенность: {confidence:.2f})",
            )

        return True, f"Категория: '{category}' (уверенность: {confidence:.2f})"
