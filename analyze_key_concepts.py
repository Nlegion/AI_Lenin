import json
import re
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
import random
import time

# Инициализация логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class LeninPhilosophyAnalyzer:
    def __init__(self, ontology_path, books_dir):
        # Оптимизированные параметры для максимального охвата
        self.MAX_OCCURRENCES = 2000  # Достаточно для репрезентативной выборки
        self.CONTEXT_SIZE = 2000  # Баланс между глубиной и производительностью
        self.EXAMPLES_TO_SAVE = 40  # Сохраняем только ключевые примеры
        self.BATCH_SIZE = 64  # Увеличенный размер пакета
        self.EMBEDDING_BATCH_SIZE = 32  # Оптимально для 8GB VRAM
        self.MIN_OCCURRENCES = 3  # Включаем редкие концепты
        self.TOP_CONCEPTS = 1000  # Топ-50 трансформированных концептов
        self.MODEL_NAME = 'all-MiniLM-L6-v2'  # Оптимальная модель

        self.ontology_path = Path(ontology_path)
        self.books_dir = Path(books_dir)
        self.device = self.select_device()
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
                "model": self.MODEL_NAME,
                "device": str(self.device)
            },
            "statistics": {
                "total_concepts": 0,
                "analyzed_concepts": 0,
                "total_occurrences": 0,
                "start_time": time.time(),
                "last_update": time.time()
            },
            "interpretations": {},
            "transformations": {},
            "top_transformed": []
        }
        logger.info("High-coverage analyzer initialized")
        logger.info(f"Target: top-{self.TOP_CONCEPTS} from {len(self.ontology)} concepts")
        logger.info(f"Using device: {self.device}")

    def select_device(self):
        """Автоматический выбор устройства с учетом памяти"""
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            logger.info(f"GPU available: {total_vram:.1f}GB total, {free_vram:.1f}GB free")

            # Используем GPU если свободно более 4GB
            if free_vram > 4.0:
                return 'cuda'
        return 'cpu'

    def load_model(self):
        """Загрузка оптимальной модели"""
        model = SentenceTransformer(
            self.MODEL_NAME,
            device=self.device,
            cache_folder='./model_cache'
        )
        model.max_seq_length = 256

        # Оптимизация для GPU
        if self.device == 'cuda':
            model = model.half()  # Используем float16 для экономии памяти
            logger.info("Model optimized for GPU (float16)")

        logger.info(f"Loaded model: {self.MODEL_NAME}")
        logger.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model

    def get_ontology_dimension(self):
        if not self.ontology:
            return 0
        first_embedding = next(iter(self.ontology.values()))['embedding']
        return len(first_embedding)

    def align_embeddings(self, embedding):
        model_dim = self.model.get_sentence_embedding_dimension()
        ontology_dim = len(embedding)

        if ontology_dim == model_dim:
            return embedding

        logger.warning(f"Embedding dimension mismatch: ontology={ontology_dim}, model={model_dim}")

        if ontology_dim > model_dim:
            return embedding[:model_dim]
        else:
            return embedding + [0.0] * (model_dim - ontology_dim)

    def load_spacy(self):
        """Загрузка облегченного NLP-пайплайна"""
        try:
            nlp = spacy.load(
                "ru_core_news_sm",
                disable=["parser", "ner", "lemmatizer", "textcat"]
            )
            logger.info("Loaded efficient Russian language processor")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            raise

        nlp.add_pipe('sentencizer')
        return nlp

    def load_ontology(self):
        if not self.ontology_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {self.ontology_path}")

        with open(self.ontology_path, 'r', encoding='utf-8') as f:
            ontology = json.load(f)

        # Фильтрация нерелевантных концептов
        normalized_ontology = {}
        if "embeddings" in ontology and isinstance(ontology["embeddings"], dict):
            for concept, embedding in ontology["embeddings"].items():
                # Пропускаем слишком короткие и числовые концепты
                if len(concept) < 2 or concept.isdigit():
                    continue

                # Приведение эмбеддингов
                aligned_embedding = self.align_embeddings(embedding)
                normalized_ontology[concept] = {"embedding": aligned_embedding}

        logger.info(f"Loaded ontology: {len(normalized_ontology)} concepts")
        return normalized_ontology

    def load_lenin_works(self):
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
                texts[work] = content
                logger.info(f"Loaded: {work} ({len(content)} chars)")

        if not texts:
            raise ValueError("No Lenin works found")

        return texts

    def build_concept_index(self):
        """Построение эффективного индекса концептов"""
        index = defaultdict(list)
        logger.info("Building concept index...")

        # Фильтрация концептов
        filtered_concepts = [c for c in self.ontology.keys() if 3 <= len(c) <= 50]

        for concept in tqdm(filtered_concepts, desc="Indexing concepts"):
            normalized = self.normalize_concept(concept)
            if normalized:
                index[normalized].append(concept)

        logger.info(f"Index built: {len(index)} normalized forms")
        return index

    def normalize_concept(self, concept):
        """Быстрая нормализация концепта"""
        # Удаление пунктуации и приведение к нижнему регистру
        cleaned = re.sub(r'[^\w\s]', '', concept)
        return cleaned.lower().strip()

    def find_concept_occurrences(self, concept):
        """Эффективный поиск вхождений с кэшированием"""
        cache_key = hashlib.md5(concept.encode('utf-8')).hexdigest()

        if cache_key in self.context_cache:
            return self.context_cache[cache_key]

        all_occurrences = []
        normalized_concept = self.normalize_concept(concept)

        if not normalized_concept:
            return [], 0

        total_count = 0
        for work_name, text in self.texts.items():
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
        """Обработка пакета концептов с управлением памятью"""
        batch_results = {}

        for concept in concept_batch:
            if concept not in self.ontology:
                continue

            concept_data = self.ontology[concept]

            # Поиск вхождений
            try:
                occurrences, total_occurrences = self.find_concept_occurrences(concept)
            except Exception as e:
                logger.error(f"Error finding occurrences for '{concept}': {str(e)}")
                continue

            # Пропуск редких концептов
            if total_occurrences < self.MIN_OCCURRENCES:
                continue

            # Выборка вхождений
            if len(occurrences) > self.MAX_OCCURRENCES:
                occurrences = random.sample(occurrences, self.MAX_OCCURRENCES)

            # Обработка вхождений пакетами
            all_embeddings = []
            for i in range(0, len(occurrences), self.EMBEDDING_BATCH_SIZE):
                batch_occurrences = occurrences[i:i + self.EMBEDDING_BATCH_SIZE]
                contexts = [occ["context"] for occ in batch_occurrences]

                try:
                    batch_embeddings = self.model.encode(
                        contexts,
                        batch_size=min(16, len(contexts)),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        device=self.device
                    )
                    # Конвертация float16 в float32 если нужно
                    if batch_embeddings.dtype == np.float16:
                        batch_embeddings = batch_embeddings.astype(np.float32)
                    all_embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error encoding batch for '{concept}': {str(e)}")
                    continue

                # Очистка памяти
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

            # Сохранение только топ-примеров
            try:
                sorted_indices = np.argsort(similarities)[::-1][:self.EXAMPLES_TO_SAVE]
                examples = []
                for idx in sorted_indices:
                    if idx < len(occurrences):
                        occ = occurrences[idx]
                        examples.append({
                            "work": occ["work"],
                            "context": occ["context"],
                            "similarity": float(similarities[idx]),
                            "transformation": float(1 - similarities[idx])
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
        """Анализ с улучшенным управлением ресурсами"""
        concepts = list(self.ontology.keys())
        total_concepts = len(concepts)
        self.report["statistics"]["total_concepts"] = total_concepts
        start_time = time.time()

        logger.info(f"Starting analysis of {total_concepts} concepts")
        logger.info(f"Target: top-{self.TOP_CONCEPTS} transformed concepts")

        # Пакетная обработка с детальным логированием
        for i in range(0, total_concepts, self.BATCH_SIZE):
            batch = concepts[i:i + self.BATCH_SIZE]
            batch_results = self.analyze_batch(batch)

            # Сохранение результатов
            for concept, data in batch_results.items():
                self.report["interpretations"][concept] = data["interpretation"]
                self.report["transformations"][concept] = data["transformation"]
                self.report["statistics"]["analyzed_concepts"] += 1
                self.report["statistics"]["total_occurrences"] += data["transformation"]["occurrences"]

            # Логирование прогресса
            elapsed = time.time() - start_time
            analyzed = self.report["statistics"]["analyzed_concepts"]
            progress = analyzed / total_concepts * 100
            speed = analyzed / (elapsed / 60)  # концептов в минуту

            logger.info(
                f"Progress: {analyzed}/{total_concepts} concepts ({progress:.1f}%) | "
                f"Speed: {speed:.1f} concepts/min | "
                f"Elapsed: {elapsed / 60:.1f} min"
            )

            # Очистка памяти
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # Анализ топ-трансформированных концептов
        self.identify_top_concepts()

        total_time = (time.time() - start_time) / 60
        logger.info("Analysis completed successfully")
        logger.info(f"Analyzed concepts: {analyzed}/{total_concepts} ({progress:.1f}%)")
        logger.info(f"Total occurrences analyzed: {self.report['statistics']['total_occurrences']}")
        logger.info(f"Total time: {total_time:.1f} minutes")

    def identify_top_concepts(self):
        """Идентификация топ-50 трансформированных концептов"""
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

        # Детальное логирование результатов
        logger.info(f"Top {self.TOP_CONCEPTS} transformed concepts:")
        for i, item in enumerate(top_transformed, 1):
            logger.info(f"{i:3d}. {item['concept'][:30]:<30} | "
                        f"Index: {item['transformation_index']:.3f} | "
                        f"Occur: {item['occurrences']}")

    def save_report(self, output_path):
        """Сохранение отчета с версионированием"""
        output_path = Path(output_path)
        version = 1
        while output_path.exists():
            backup_path = output_path.with_name(f"{output_path.stem}_v{version}{output_path.suffix}")
            if not backup_path.exists():
                break
            version += 1

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)

        logger.info(f"Report saved to {output_path}")
        logger.info(f"Report size: {os.path.getsize(output_path) / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    # Конфигурация
    PROJECT_ROOT = Path(__file__).parent
    ONTOLOGY_PATH = PROJECT_ROOT / "foundation_ontology.json"
    BOOKS_DIR = PROJECT_ROOT / "books"
    OUTPUT_PATH = PROJECT_ROOT / "lenin_dialectical_specificity_top50.json"

    # Очистка памяти перед запуском
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        logger.info("Starting Top-50 Lenin Philosophy Analyzer")
        logger.info("Hardware configuration:")
        logger.info(f"  CPU: AMD Ryzen 7 5800X")
        logger.info(f"  RAM: 64GB (49GB available)")
        logger.info(f"  GPU: RTX 4060 8GB")
        logger.info("Analysis parameters:")
        logger.info(f"  Max context: {2000} characters")
        logger.info(f"  Max samples per concept: {2000}")
        logger.info(f"  Batch size: {64} concepts")
        logger.info(f"  Embedding batch size: {32}")
        logger.info(f"  Top concepts: {50}")

        analyzer = LeninPhilosophyAnalyzer(ONTOLOGY_PATH, BOOKS_DIR)
        analyzer.analyze()
        analyzer.save_report(OUTPUT_PATH)
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.exception(f"Analysis failed: {str(e)}")