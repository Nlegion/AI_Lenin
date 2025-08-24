import os
import json
import spacy
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import re
import logging
import time
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ontology_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OntologyBuilder")


class OntologyBuilder:
    def __init__(self):
        logger.info("Инициализация NLP модели...")
        self.nlp = spacy.load("ru_core_news_lg")
        self.nlp.max_length = 5000000

        logger.info("Загрузка модели эмбеддингов...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.graph = nx.Graph()
        self.concept_embeddings = {}
        self.concept_relations = defaultdict(lambda: defaultdict(float))
        self.philosophy_terms = self.load_philosophy_terms()
        logger.info("Инициализация завершена")

    def load_philosophy_terms(self):
        """Расширенный философский тезаурус с биграммами"""
        logger.info("Загрузка философского тезауруса")
        return {
            # Основные философские направления
            "диалектика": [
                "диалектический метод", "единство противоположностей",
                "отрицание отрицания", "переход количества в качество",
                "борьба противоположностей", "диалектическое развитие"
            ],
            "материализм": [
                "исторический материализм", "диалектический материализм",
                "материалистическое понимание истории", "теория отражения",
                "объективная реальность", "примат материи"
            ],
            "идеализм": [
                "объективный идеализм", "субъективный идеализм",
                "абсолютная идея", "мировой разум"
            ],
            "познание": [
                "гносеология", "теория познания", "чувственное познание",
                "логическое познание", "практика как критерий истины",
                "субъективный образ объективной реальности"
            ],
            "революция": [
                "классовая борьба", "диктатура пролетариата",
                "социалистическая революция", "революционная ситуация",
                "авангард партии", "насильственное свержение"
            ],
            "онтология": [
                "бытие", "сущность", "материя", "субстанция",
                "атрибут", "модус", "пространство и время"
            ],
            "экономика": [
                "производительные силы", "производственные отношения",
                "способ производства", "базис и надстройка",
                "прибавочная стоимость", "эксплуатация"
            ],

            # Конкретные концепты
            "пролетариат": ["рабочий класс", "индустриальные рабочие"],
            "буржуазия": ["капиталистический класс", "эксплуататорский класс"],
            "империализм": [
                "высшая стадия капитализма", "монополистический капитализм",
                "финансовый капитал", "экспорт капитала"
            ],
            "государство": [
                "машина подавления", "орган классового господства",
                "диктатура класса", "отмирание государства"
            ],
            "партия": [
                "авангард рабочего класса", "революционная организация",
                "демократический централизм", "партийная дисциплина"
            ],

            # Философы и их концепции
            "Гегель": ["абсолютная идея", "диалектика Гегеля"],
            "Маркс": ["марксизм", "учение Маркса"],
            "Ленин": ["ленинизм", "теория Ленина"],
            "Фейербах": ["антропологизм", "критика религии"],
            "Кант": ["категорический императив", "вещь в себе"],
            "Аристотель": ["форма и материя", "четыре причины"]
        }

    def extract_key_concepts(self, text):
        if not text:
            logger.warning("Получен пустой текст")
            return [], []

        logger.info(f"Начало обработки текста длиной {len(text)} символов")
        start_time = time.time()
        all_concepts = set()

        # Извлечение только ключевых терминов без разбивки на фрагменты
        try:
            # Философские термины
            for term, synonyms in self.philosophy_terms.items():
                if term in text.lower():
                    all_concepts.add(term)
                for synonym in synonyms:
                    if synonym in text.lower():
                        all_concepts.add(term)

            # Именованные сущности (только люди и работы)
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PER", "WORK_OF_ART"]:
                    all_concepts.add(ent.text)
        except Exception as e:
            logger.error(f"Ошибка обработки текста: {str(e)}")

        concepts = list(all_concepts)
        logger.info(f"Извлечено {len(concepts)} концептов из текста")

        if concepts:
            logger.info("Создание эмбеддингов...")
            embeddings = self.model.encode(concepts)
            logger.info(f"Обработка текста завершена за {time.time() - start_time:.2f} сек.")
            return concepts, embeddings

        return [], []

    def build_concept_relations(self, concepts, embeddings):
        if not concepts:
            return

        logger.info(f"Построение отношений для {len(concepts)} концептов")
        start_time = time.time()

        # Фильтрация концептов
        filtered_concepts = []
        filtered_embeddings = []
        for concept, emb in zip(concepts, embeddings):
            if 3 <= len(concept) <= 50:
                filtered_concepts.append(concept)
                filtered_embeddings.append(emb)

        if not filtered_concepts:
            return

        # Векторизованное вычисление сходства
        emb_matrix = np.array(filtered_embeddings)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        normalized_emb = emb_matrix / norms
        similarity_matrix = np.dot(normalized_emb, normalized_emb.T)

        # Добавление связей
        relations_count = 0
        for i in range(len(filtered_concepts)):
            concept1 = filtered_concepts[i]
            self.concept_embeddings[concept1] = filtered_embeddings[i]

            for j in range(i + 1, len(filtered_concepts)):
                if similarity_matrix[i, j] > 0.6:
                    concept2 = filtered_concepts[j]
                    self.concept_relations[concept1][concept2] = float(similarity_matrix[i, j])
                    relations_count += 1

        logger.info(f"Добавлено {relations_count} отношений за {time.time() - start_time:.2f} сек.")

    def save_ontology(self, output_file):
        logger.info(f"Сохранение онтологии в {output_file}")
        ontology = {
            "concepts": list(self.concept_embeddings.keys()),
            "relations": [
                {
                    "source": source,
                    "target": target,
                    "strength": strength
                }
                for source, targets in self.concept_relations.items()
                for target, strength in targets.items()
            ],
            "embeddings": {
                concept: embedding.tolist()
                for concept, embedding in self.concept_embeddings.items()
            }
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, ensure_ascii=False, indent=2)
        logger.info("Онтология успешно сохранена")


def load_processed_texts(directory):
    logger.info(f"Загрузка текстов из {directory}")
    texts = []

    if not os.path.exists(directory):
        return texts

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                try:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'content' in data:
                            texts.append(data['content'])
                except:
                    continue

    return texts


if __name__ == "__main__":
    logger.info("Начало построения онтологии")
    builder = OntologyBuilder()

    # Обрабатываем философов
    philosophers = ["Гегель", "Маркс", "Фейербах"]
    for philosopher in philosophers:
        dir_path = os.path.join("data/books/processed_intellectual", philosopher)
        if not os.path.exists(dir_path):
            continue

        logger.info(f"Обработка философа: {philosopher}")
        texts = load_processed_texts(dir_path)

        for text in texts:
            concepts, embeddings = builder.extract_key_concepts(text)
            if concepts:
                builder.build_concept_relations(concepts, embeddings)

    # Сохранение результатов
    builder.save_ontology("foundation_ontology.json")
    logger.info("Онтология успешно построена и сохранена")