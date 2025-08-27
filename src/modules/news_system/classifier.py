import re
import logging
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)


class NewsClassifier:
    def __init__(self):
        # Обновленные категории и ключевые слова
        self.categories = {
            "politics": ["политика", "правительство", "выборы", "партия", "власть",
                         "государство", "президент", "министр", "кремль", "санкции"],
            "economics": ["экономика", "финансы", "рынок", "бизнес", "компания",
                          "капитал", "инвестиции", "банк", "валюта", "бюджет"],
            "social": ["общество", "социальный", "труд", "работа", "зарплата",
                       "пенсия", "медицина", "образование", "здравоохранение"],
            "international": ["международный", "дипломатия", "ООН", "НАТО", "ЕС",
                              "санкции", "посол", "договор", "переговоры"],
            "culture": ["культура", "искусство", "кино", "театр", "литература",
                        "музей", "выставка", "концерт", "фестиваль"],
            "other": ["технология", "спорт", "развлечение", "наука"]
        }

        # Приоритетные категории для анализа
        self.high_priority_categories = ["politics", "economics", "social", "international"]
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
                "politics": ["заседание правительства", "встреча с президентом",
                             "политическое решение", "государственная программа"],
                "economics": ["финансовый отчет", "экономический рост",
                              "рыночные показатели", "бизнес проект"],
                "social": ["социальная защита", "трудовые отношения",
                           "медицинское обслуживание", "образовательная программа"],
                "international": ["международные отношения", "дипломатическая миссия",
                                  "встреча на высшем уровне", "международный договор"],
                "culture": ["культурное мероприятие", "художественная выставка",
                            "театральная постановка", "литературный вечер"]
            }

            if category in context_examples:
                for example in context_examples[category]:
                    texts.append(example)
                    labels.append(category)

        # Создаем и обучаем модель
        model = make_pipeline(
            TfidfVectorizer(max_features=1000, ngram_range=(1, 2)),
            MultinomialNB(alpha=0.1)
        )
        model.fit(texts, labels)

        return model

    def classify_news(self, title: str, content: str) -> Tuple[str, float]:
        """Улучшенная классификация новости"""
        # Объединяем заголовок и содержание
        text = f"{title} {content}".lower()

        # Удаляем стоп-слова и короткие слова
        words = re.findall(r'\b\w{4,}\b', text)
        processed_text = ' '.join(words)

        # Предсказание категории
        try:
            category = self.model.predict([processed_text])[0]
            probability = max(self.model.predict_proba([processed_text])[0])
            return category, probability
        except:
            # В случае ошибки возвращаем категорию по умолчанию
            return "other", 0.5

    def should_analyze(self, title: str, content: str) -> Tuple[bool, str]:
        """Определяет, нужно ли анализировать новость"""
        category, confidence = self.classify_news(title, content)

        # Понижаем пороги для приоритетных категорий
        if category in self.high_priority_categories:
            if confidence < 0.3:  # Было 0.4
                return False, f"Пропуск: низкая уверенность в категории '{category}' (уверенность: {confidence:.2f})"
            return True, f"Категория: '{category}' (уверенность: {confidence:.2f})"

        if category in self.low_priority_categories:
            if confidence < 0.5:  # Было 0.6
                return False, f"Пропуск: низкая уверенность в категории '{category}' (уверенность: {confidence:.2f})"
            return True, f"Категория: '{category}' (уверенность: {confidence:.2f})"

        if category in self.skip_categories:
            return False, f"Пропуск: категория '{category}' (уверенность: {confidence:.2f})"

        # Для неизвестных категорий используем эвристический анализ
        text = f"{title} {content}".lower()
        political_terms = ["политика", "правительство", "президент", "выборы"]
        economic_terms = ["экономика", "финансы", "бизнес", "рынок"]

        political_score = sum(1 for term in political_terms if term in text)
        economic_score = sum(1 for term in economic_terms if term in text)

        if political_score >= 2 or economic_score >= 2:
            return True, f"Эвристический анализ: политика={political_score}, экономика={economic_score}"

        return False, f"Пропуск: неопределенная категория '{category}' (уверенность: {confidence:.2f})"