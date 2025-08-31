import json
import re
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
import spacy
import torch
import os
import gc
from collections import defaultdict
import hashlib
import psutil

# Инициализация логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Установка переменной окружения для управления памятью CUDA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class LeninPhilosophyAnalyzer:
    def __init__(self, ontology_path, books_dir):
        # Параметры для работы на CPU с ограниченной памятью
        self.MAX_OCCURRENCES = 500  # Значительно уменьшено
        self.CONTEXT_SIZE = 1000  # Уменьшенный контекст
        self.EXAMPLES_TO_SAVE = 50  # Меньше примеров
        self.BATCH_SIZE = 16  # Маленькие пакеты
        self.EMBEDDING_BATCH_SIZE = 4  # Очень маленькие пакеты для эмбеддингов
        self.MIN_OCCURRENCES = 5  # Только концепты с достаточным количеством вхождений
        self.TOP_CONCEPTS = 10  # Меньше топ-концептов

        self.ontology_path = Path(ontology_path)
        self.books_dir = Path(books_dir)
        self.device = self.select_device()  # Автоматический выбор устройства
        self.model = self.load_model()
        self.ontology = self.load_ontology()
        self.nlp = self.load_spacy()
        self.texts = self.load_lenin_works()
        self.concept_index = self.build_concept_index()
        self.context_cache = {}
        self.report = {
            "parameters": {
                "max_occurrences": self.MAX_OCCURRENCES,
                "context_size": self.CONTEXT_SIZE,
                "examples_to_save": self.EXAMPLES_TO_SAVE,
                "batch_size": self.BATCH_SIZE,
                "min_occurrences": self.MIN_OCCURRENCES,
                "top_concepts": self.TOP_CONCEPTS,
                "device": str(self.device)
            },
            "statistics": {
                "total_concepts": 0,
                "analyzed_concepts": 0,
                "total_occurrences": 0
            },
            "interpretations": {},
            "transformations": {},
            "top_transformed": []
        }
        logger.info("Memory-safe analyzer initialized")
        logger.info(f"Optimized parameters: context={self.CONTEXT_SIZE} chars, max_samples={self.MAX_OCCURRENCES}")
        logger.info(f"Using device: {self.device}")

    def select_device(self):
        """Выбор устройства в зависимости от доступной памяти"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_memory = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # Свободная память в GB
            if free_memory > 2:  # Только если доступно более 2GB GPU памяти
                return 'cuda'
        return 'cpu'

    def load_model(self):
        """Загрузка легкой модели, оптимизированной для CPU"""
        # Используем меньшую модель для экономии памяти
        model_name = 'all-MiniLM-L6-v2'
        model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder='./model_cache'
        )
        model.max_seq_length = 256  # Уменьшенная длина последовательности

        # Дополнительная оптимизация для CPU
        if self.device == 'cpu':
            model = model.to('cpu')
            torch.set_num_threads(psutil.cpu_count(logical=False))  # Используем физические ядра

        logger.info(f"Loaded optimized model: {model_name}")
        logger.info(f"Model embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model

    def get_ontology_dimension(self):
        """Получение размерности эмбеддингов онтологии"""
        if not self.ontology:
            return 0
        first_embedding = next(iter(self.ontology.values()))['embedding']
        return len(first_embedding)

    def align_embeddings(self, embedding):
        """Приведение эмбеддингов к единой размерности"""
        model_dim = self.model.get_sentence_embedding_dimension()
        ontology_dim = len(embedding)

        if ontology_dim == model_dim:
            return embedding

        logger.warning(f"Embedding dimension mismatch: ontology={ontology_dim}, model={model_dim}")

        # Простое дополнение или обрезка
        if ontology_dim > model_dim:
            return embedding[:model_dim]
        else:
            return embedding + [0.0] * (model_dim - ontology_dim)

    def load_spacy(self):
        """Загрузка NLP-пайплайна с оптимизацией памяти"""
        # Использование самой легкой модели
        try:
            nlp = spacy.load(
                "ru_core_news_sm",  # Маленькая модель
                disable=["parser", "ner", "lemmatizer", "textcat"]
            )
            logger.info("Loaded small Russian language processor")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            raise

        nlp.add_pipe('sentencizer')
        return nlp

    def load_ontology(self):
        """Загрузка и нормализация онтологии"""
        if not self.ontology_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {self.ontology_path}")

        with open(self.ontology_path, 'r', encoding='utf-8') as f:
            ontology = json.load(f)

        # Оптимизированная обработка онтологии
        normalized_ontology = {}
        if "embeddings" in ontology and isinstance(ontology["embeddings"], dict):
            for concept, embedding in ontology["embeddings"].items():
                # Фильтрация неинформативных концептов
                if len(concept) < 3 or concept.isdigit():
                    continue

                # Приведение эмбеддингов к размерности модели
                aligned_embedding = self.align_embeddings(embedding)
                normalized_ontology[concept] = {"embedding": aligned_embedding}

        logger.info(f"Optimized ontology loaded: {len(normalized_ontology)} concepts")
        return normalized_ontology

    def load_lenin_works(self):
        """Загрузка и предобработка текстов"""
        texts = {}
        lenin_dir = self.books_dir / "Ленин"

        if not lenin_dir.exists():
            raise FileNotFoundError(f"Lenin works directory not found: {lenin_dir}")

        works = ["Философские тетради.txt", "Материализм и эмпириокритицизм.txt"]
        for work in works:
            work_path = lenin_dir / work
            if work_path.exists():
                with open(work_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Базовая предобработка текста
                content = self.preprocess_text(content)
                texts[work] = content
                logger.info(f"Loaded: {work} ({len(content)} chars)")

        if not texts:
            raise ValueError("No Lenin works found")

        return texts

    def preprocess_text(self, text):
        """Базовая предобработка текста"""
        # Удаление технических артефактов
        text = re.sub(r'\ufeff', '', text)  # Удаление BOM
        text = re.sub(r'\r\n', '\n', text)  # Нормализация переносов
        return text

    def build_concept_index(self):
        """Создание оптимизированного индекса концептов"""
        index = defaultdict(list)
        logger.info("Building optimized concept index...")

        # Фильтрация концептов для уменьшения нагрузки
        filtered_concepts = [c for c in self.ontology.keys() if 3 <= len(c) <= 50]

        for concept in tqdm(filtered_concepts, desc="Indexing concepts"):
            # Базовая нормализация
            normalized = self.normalize_concept(concept)
            if normalized:
                index[normalized].append(concept)

        logger.info(f"Optimized index built: {len(index)} normalized forms")
        return index

    def normalize_concept(self, concept):
        """Базовая нормализация концепта"""
        # Удаление пунктуации
        cleaned = re.sub(r'[^\w\s]', '', concept)
        return cleaned.lower().strip()

    def find_concept_occurrences(self, concept):
        """Оптимизированный поиск вхождений"""
        # Генерация ключа кэша
        cache_key = hashlib.md5(concept.encode('utf-8')).hexdigest()

        # Использование кэшированных результатов
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]

        all_occurrences = []
        normalized_concept = self.normalize_concept(concept)

        if not normalized_concept:
            return [], 0

        # Простой поиск с учетом контекста
        total_count = 0
        for work_name, text in self.texts.items():
            # Простой поиск без регулярок для экономии памяти
            start_idx = 0
            while start_idx < len(text):
                pos = text.lower().find(normalized_concept, start_idx)
                if pos == -1:
                    break

                end_idx = pos + len(normalized_concept)
                context_start = max(0, pos - self.CONTEXT_SIZE)
                context_end = min(len(text), end_idx + self.CONTEXT_SIZE)
                context = text[context_start:context_end]

                all_occurrences.append({
                    "work": work_name,
                    "context": context,
                    "position": (pos, end_idx),
                    "match": text[pos:end_idx]
                })

                total_count += 1
                start_idx = end_idx

                # Остановка при достижении лимита
                if len(all_occurrences) >= self.MAX_OCCURRENCES * 2:
                    break

        # Кэширование результатов
        self.context_cache[cache_key] = (all_occurrences, total_count)
        return all_occurrences, total_count

    def calculate_transformation(self, original_embedding, interpretation_embeddings):
        """Оптимизированный расчет метрик"""
        original_embedding = np.array(original_embedding, dtype=np.float32)
        interpretation_embeddings = np.array(interpretation_embeddings, dtype=np.float32)

        # Проверка размерностей
        if original_embedding.shape[0] != interpretation_embeddings.shape[1]:
            raise ValueError(
                f"Dimension mismatch: original ({original_embedding.shape[0]}), "
                f"interpretations ({interpretation_embeddings.shape[1]})"
            )

        # Нормализация
        original_norm = np.linalg.norm(original_embedding)
        if original_norm == 0:
            return 1.0, [0.0] * len(interpretation_embeddings)

        original_embedding /= original_norm
        norms = np.linalg.norm(interpretation_embeddings, axis=1)
        norms[norms == 0] = 1e-10
        interpretation_embeddings = interpretation_embeddings / norms[:, np.newaxis]

        # Расчет сходства
        similarities = np.dot(interpretation_embeddings, original_embedding)
        avg_similarity = np.mean(similarities)

        return 1 - avg_similarity, similarities.tolist()

    def analyze_batch(self, concept_batch):
        """Оптимизированная обработка пакета концептов"""
        batch_results = {}

        for concept in concept_batch:
            if concept not in self.ontology:
                continue

            concept_data = self.ontology[concept]

            # Поиск вхождений
            occurrences, total_occurrences = self.find_concept_occurrences(concept)

            # Пропуск редких концептов
            if total_occurrences < self.MIN_OCCURRENCES:
                continue

            # Выборка вхождений
            if len(occurrences) > self.MAX_OCCURRENCES:
                # Простая случайная выборка
                occurrences = random.sample(occurrences, self.MAX_OCCURRENCES)

            # Обработка вхождений очень маленькими пакетами
            all_embeddings = []
            for i in range(0, len(occurrences), self.EMBEDDING_BATCH_SIZE):
                batch_occurrences = occurrences[i:i + self.EMBEDDING_BATCH_SIZE]
                contexts = [occ["context"] for occ in batch_occurrences]

                # Создание эмбеддингов для подпакета
                try:
                    batch_embeddings = self.model.encode(
                        contexts,
                        batch_size=2,  # Очень маленький размер пакета
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        device=self.device
                    )
                    all_embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error encoding batch for '{concept}': {str(e)}")
                    continue

                # Очистка памяти после каждого подпакета
                del batch_embeddings
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            if not all_embeddings:
                continue

            context_embeddings = np.vstack(all_embeddings)

            # Расчет метрик
            try:
                transformation_index, similarities = self.calculate_transformation(
                    concept_data['embedding'],
                    context_embeddings
                )
            except Exception as e:
                logger.error(f"Error calculating transformation for '{concept}': {str(e)}")
                continue

            # Отбор примеров
            try:
                sorted_indices = np.argsort(similarities)[::-1]
                example_indices = sorted_indices[:self.EXAMPLES_TO_SAVE]
                examples = []
                for i in example_indices:
                    if i < len(occurrences):
                        occ = occurrences[i]
                        examples.append({
                            "work": occ["work"],
                            "context": occ["context"],
                            "similarity": float(similarities[i]),
                            "transformation": float(1 - similarities[i])
                        })
            except Exception as e:
                logger.error(f"Error selecting examples for '{concept}': {str(e)}")
                examples = []

            # Сохранение результатов
            batch_results[concept] = {
                "interpretation": {
                    "original_embedding": [float(x) for x in concept_data['embedding']],
                    "transformation_index": float(transformation_index),
                    "sampled_occurrences": len(occurrences),
                    "total_occurrences": total_occurrences,
                    "examples": examples,
                },
                "transformation": {
                    "transformation_index": float(transformation_index),
                    "occurrences": total_occurrences
                }
            }

            # Освобождение памяти
            del context_embeddings, all_embeddings
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        return batch_results

    def analyze(self):
        """Безопасный анализ с управлением памятью"""
        # Подготовка концептов
        concepts = list(self.ontology.keys())
        total_concepts = len(concepts)
        self.report["statistics"]["total_concepts"] = total_concepts

        logger.info(f"Starting safe analysis of {total_concepts} concepts")

        # Пакетная обработка с прогресс-баром
        analyzed = 0
        for i in range(0, total_concepts, self.BATCH_SIZE):
            batch = concepts[i:i + self.BATCH_SIZE]
            batch_results = self.analyze_batch(batch)
            analyzed += len(batch)

            # Сохранение результатов
            for concept, data in batch_results.items():
                self.report["interpretations"][concept] = data["interpretation"]
                self.report["transformations"][concept] = data["transformation"]
                self.report["statistics"]["analyzed_concepts"] += 1
                self.report["statistics"]["total_occurrences"] += data["transformation"]["occurrences"]

            # Логирование прогресса
            logger.info(
                f"Progress: {analyzed}/{total_concepts} concepts, {self.report['statistics']['analyzed_concepts']} analyzed")

            # Очистка памяти после каждого пакета
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # Анализ топ-трансформированных концептов
        self.identify_top_concepts()

        logger.info("Safe analysis completed successfully")
        logger.info(f"Analyzed concepts: {self.report['statistics']['analyzed_concepts']}/{total_concepts}")
        logger.info(f"Total occurrences analyzed: {self.report['statistics']['total_occurrences']}")

    def identify_top_concepts(self):
        """Идентификация наиболее трансформированных концептов"""
        transformed_concepts = []
        for concept, data in self.report["transformations"].items():
            if data["occurrences"] >= self.MIN_OCCURRENCES:
                transformed_concepts.append({
                    "concept": concept,
                    "transformation_index": data["transformation_index"],
                    "occurrences": data["occurrences"]
                })

        # Сортировка и выбор топ-концептов
        top_transformed = sorted(
            transformed_concepts,
            key=lambda x: x["transformation_index"],
            reverse=True
        )[:self.TOP_CONCEPTS]

        self.report["top_transformed"] = top_transformed

        # Логирование топ-результатов
        logger.info(f"Top {self.TOP_CONCEPTS} transformed concepts:")
        for i, item in enumerate(top_transformed, 1):
            logger.info(f"{i:2d}. {item['concept'][:30]:<30} | "
                        f"Index: {item['transformation_index']:.3f} | "
                        f"Occur: {item['occurrences']}")

    def save_report(self, output_path):
        """Сохранение отчета"""
        output_path = Path(output_path)

        # Сохранение отчета
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)

        logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    # Конфигурация
    PROJECT_ROOT = Path(__file__).parent
    ONTOLOGY_PATH = PROJECT_ROOT / "foundation_ontology.json"
    BOOKS_DIR = PROJECT_ROOT / "books"
    OUTPUT_PATH = PROJECT_ROOT / "lenin_dialectical_specificity_safe.json"

    # Очистка памяти перед запуском
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        logger.info("Starting Safe Lenin Philosophy Analyzer")
        logger.info("Optimized configuration:")
        logger.info(f"  Max context: {1000} characters")
        logger.info(f"  Max samples per concept: {500}")
        logger.info(f"  Batch size: {16} concepts")
        logger.info(f"  Embedding batch size: {4}")
        logger.info(f"  Top concepts: {10}")

        analyzer = LeninPhilosophyAnalyzer(ONTOLOGY_PATH, BOOKS_DIR)
        analyzer.analyze()
        analyzer.save_report(OUTPUT_PATH)

        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.exception(f"Analysis failed: {str(e)}")