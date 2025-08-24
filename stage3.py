import os
import json
import re
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
import logging
import datetime
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("worldview_builder.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


class HistoricalContext:
    """Углубленный исторический контекст с интеграцией первых двух этапов"""

    def __init__(self, stage1_path, stage2_path):
        self.key_events = self.load_events()
        self.ontological_maps = self.load_ontological_maps(stage1_path)
        self.transformation_patterns = self.load_transformation_patterns(stage2_path)

    def load_events(self):
        """Ключевые исторические события с их влиянием"""
        return {
            1893: {"event": "Начало революционной деятельности", "impact": "Формирование марксистских взглядов"},
            1900: {"event": "Основание 'Искры'", "impact": "Развитие организационных принципов"},
            1905: {"event": "Первая русская революция", "impact": "Теория революционной ситуации"},
            1914: {"event": "Начало Первой мировой войны", "impact": "Теория империализма"},
            1917: {"event": "Февральская революция", "impact": "Апрельские тезисы"},
            1917: {"event": "Октябрьская революция", "impact": "Теория государства и революции"},
            1921: {"event": "НЭП", "impact": "Пересмотр экономической политики"}
        }

    def load_ontological_maps(self, path):
        """Загрузка онтологических карт из Этапа 1"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'concepts': data.get('concepts', []),
                    'relations': data.get('relations', []),
                    'embeddings': data.get('embeddings', {})
                }
        except Exception as e:
            logger.error(f"Не удалось загрузить онтологические карты: {str(e)}")
            return {
                'concepts': [],
                'relations': [],
                'embeddings': {}
            }

    def load_transformation_patterns(self, path):
        """Загрузка паттернов трансформации из Этапа 2"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'transformations': data.get('transformations', {}),
                    'interpretations': data.get('interpretations', {})
                }
        except Exception as e:
            logger.error(f"Не удалось загрузить паттерны трансформации: {str(e)}")
            return {
                'transformations': {},
                'interpretations': {}
            }

    def get_context_for_year(self, year):
        """Получение исторического контекста для конкретного года"""
        events = [e for y, e in self.key_events.items() if y <= year]
        return sorted(events, key=lambda x: list(self.key_events.keys())[list(self.key_events.values()).index(x)])

    def get_ontological_foundation(self, concept):
        """Получение онтологических связей для концепта"""
        for relation in self.ontological_maps['relations']:
            if relation['source'] == concept or relation['target'] == concept:
                return relation
        return {}

    def get_transformation_index(self, concept):
        """Получение индекса трансформации для концепта"""
        return self.transformation_patterns['transformations'].get(concept, {}).get("transformation_index", 1.0)


class TextPreprocessor:
    """Улучшенная предварительная обработка с учетом исторического контекста"""

    def __init__(self, historical_context):
        self.historical_context = historical_context
        self.stop_words = self.load_stop_words()
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    def load_stop_words(self):
        return set([
            "и", "в", "не", "на", "с", "по", "к", "а", "из", "от", "то",
            "что", "как", "но", "он", "я", "мы", "вы", "его", "ее", "их",
            "это", "тот", "этот", "такой", "где", "когда", "даже", "лишь",
            "уже", "или", "если", "чтобы", "хотя", "за", "до", "после"
        ])

    def clean_text(self, text):
        text = re.sub(r"[^а-яА-ЯёЁ\s\-]", "", text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        tokens = text.split()
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]

    def preprocess(self, text):
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return tokens

    def contextual_embedding(self, text, year):
        """Семантическое кодирование с учетом исторического контекста"""
        context = self.historical_context.get_context_for_year(year)
        context_str = " ".join([e["event"] for e in context])
        combined_text = f"{context_str} {text}"
        return self.model.encode(combined_text)


class StylometricAnalyzer:
    """Анализ стилистических особенностей текста"""

    def __init__(self):
        self.function_words = [
            "и", "в", "не", "на", "с", "по", "к", "а", "из", "от", "то",
            "что", "как", "но", "же", "бы", "вот", "ли", "только", "уже"
        ]

    def analyze(self, text):
        """Многоуровневый стилометрический анализ"""
        # Разделение на предложения
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

        # Разделение на слова
        words = text.split()

        if not words:
            return {}

        # Подсчет уникальных слов
        unique_words = set(words)

        # Расчет метрик
        return {
            'sentence_length': self.avg_sentence_length(sentences),
            'lexical_diversity': len(unique_words) / len(words),
            'function_words': self.function_word_frequency(words),
            'punctuation': self.punctuation_analysis(text),
            'paragraph_length': self.avg_paragraph_length(text),
            'word_length': np.mean([len(word) for word in words])
        }

    def avg_sentence_length(self, sentences):
        """Средняя длина предложения в словах"""
        if not sentences:
            return 0
        return np.mean([len(sentence.split()) for sentence in sentences])

    def function_word_frequency(self, words):
        """Частота использования служебных слов"""
        total = len(words)
        if total == 0:
            return {}
        counts = Counter(words)
        return {fw: counts.get(fw, 0) / total for fw in self.function_words}

    def punctuation_analysis(self, text):
        """Анализ использования пунктуации"""
        return {
            'comma': text.count(','),
            'semicolon': text.count(';'),
            'colon': text.count(':'),
            'dash': text.count('—'),
            'quote': text.count('"') + text.count("'"),
            'parenthesis': text.count('(') + text.count(')')
        }

    def avg_paragraph_length(self, text):
        """Средняя длина абзаца в предложениях"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return 0
        return np.mean([len(re.split(r'[.!?]', p)) for p in paragraphs])

    def compare(self, profile1, profile2):
        """Вычисление стилометрического сходства"""
        # Весовые коэффициенты
        weights = {
            'sentence_length': 0.25,
            'lexical_diversity': 0.20,
            'function_words': 0.30,
            'word_length': 0.15,
            'paragraph_length': 0.10
        }

        # Сравнение по каждому параметру
        scores = {
            'sentence_length': 1 - abs(profile1['sentence_length'] - profile2['sentence_length']) / 50,
            'lexical_diversity': 1 - abs(profile1['lexical_diversity'] - profile2['lexical_diversity']),
            'function_words': self.compare_function_words(
                profile1['function_words'],
                profile2['function_words']
            ),
            'word_length': 1 - abs(profile1['word_length'] - profile2['word_length']) / 3,
            'paragraph_length': 1 - abs(profile1['paragraph_length'] - profile2['paragraph_length']) / 5
        }

        # Общий взвешенный балл
        total_score = sum(scores[k] * weights[k] for k in weights)
        return min(1.0, max(0.0, total_score))

    def compare_function_words(self, fw1, fw2):
        """Сравнение частот служебных слов"""
        total_similarity = 0
        count = 0

        for word in set(fw1.keys()) | set(fw2.keys()):
            freq1 = fw1.get(word, 0)
            freq2 = fw2.get(word, 0)
            similarity = 1 - abs(freq1 - freq2)
            total_similarity += similarity
            count += 1

        return total_similarity / count if count > 0 else 0


class ConceptualNetworkBuilder:
    """Улучшенное построение семантических сетей с интеграцией первых этапов"""

    def __init__(self, preprocessor, historical_context):
        self.preprocessor = preprocessor
        self.historical_context = historical_context
        self.vectorizer = TfidfVectorizer(max_features=2000)
        self.svd = TruncatedSVD(n_components=50)

    def extract_key_terms(self, text, year, n=20):
        """Извлечение терминов с учетом онтологических связей"""
        try:
            tokens = self.preprocessor.preprocess(text)
            preprocessed_text = " ".join(tokens)

            if not preprocessed_text.strip():
                return []

            vectorizer = TfidfVectorizer(max_features=n * 2) if len(tokens) > 100 else CountVectorizer(
                max_features=n * 2)
            vectorizer.fit([preprocessed_text])
            terms = vectorizer.get_feature_names_out()

            # Ранжирование терминов с учетом онтологической значимости
            ranked_terms = []
            for term in terms:
                # Учет трансформации концепта из Этапа 2
                transformation = self.historical_context.get_transformation_index(term)
                # Учет онтологических связей из Этапа 1
                ontological_links = self.historical_context.get_ontological_foundation(term)
                ontological_score = len(ontological_links) if ontological_links else 0
                score = transformation * (1 + ontological_score * 0.1)
                ranked_terms.append((term, score))

            # Сортировка по значимости
            ranked_terms.sort(key=lambda x: x[1], reverse=True)
            return [term for term, score in ranked_terms[:n]]

        except Exception as e:
            logger.error(f"Ошибка извлечения терминов: {str(e)}")
            return []

    def build_network(self, texts, dates, author_labels=None):
        """Построение сети с историческим контекстом"""
        logger.info("Построение концептуальной сети с историческим контекстом")
        try:
            graph = nx.Graph()

            # Добавление узлов (авторов/текстов) с историческим контекстом
            for i, (text, year) in enumerate(zip(texts, dates)):
                author = author_labels[i] if author_labels else f"text_{i}"
                context = self.historical_context.get_context_for_year(year)

                # Добавляем исторический контекст как атрибуты узла
                graph.add_node(author, type="text", year=year, context=context,
                               author=author_labels[i] if author_labels else "unknown")

                # Извлечение терминов с учетом исторического контекста
                text_terms = self.extract_key_terms(text, year, n=15)
                for term in text_terms:
                    if term not in graph:
                        # Добавляем онтологические связи из Этапа 1
                        ontological = self.historical_context.get_ontological_foundation(term)
                        # Добавляем индекс трансформации из Этапа 2
                        transformation = self.historical_context.get_transformation_index(term)
                        graph.add_node(term, type="concept", ontological=ontological,
                                       transformation=transformation)

                    # Усиливаем связь для трансформированных концептов
                    weight = 1.0 * graph.nodes[term].get('transformation', 1.0)
                    graph.add_edge(author, term, weight=weight)

            # Построение связей между терминами
            self.build_term_connections(graph, texts, dates)

            return graph

        except Exception as e:
            logger.error(f"Ошибка построения сети: {str(e)}")
            return nx.Graph()

    def build_term_connections(self, graph, texts, dates):
        """Построение связей между терминами с эволюционным учетом"""
        term_cooccurrence = defaultdict(lambda: defaultdict(int))
        term_evolution = defaultdict(lambda: defaultdict(list))

        for text, year in zip(texts, dates):
            text_terms = self.extract_key_terms(text, year, n=30)
            for i, term1 in enumerate(text_terms):
                for term2 in text_terms[i + 1:]:
                    term_cooccurrence[term1][term2] += 1
                    term_cooccurrence[term2][term1] += 1
                    term_evolution[term1][term2].append(year)
                    term_evolution[term2][term1].append(year)

        for term1, co_terms in term_cooccurrence.items():
            for term2, count in co_terms.items():
                if term1 in graph and term2 in graph:
                    # Учет эволюции связи
                    years = term_evolution[term1][term2]
                    duration = max(years) - min(years) if years else 0
                    stability = 1 + duration * 0.01

                    # Учет онтологических связей
                    ontological_match = self.calculate_ontological_similarity(
                        graph.nodes[term1].get('ontological', {}),
                        graph.nodes[term2].get('ontological', {})
                    )

                    weight = count * stability * ontological_match
                    graph.add_edge(term1, term2, weight=weight, years=years)

    def calculate_ontological_similarity(self, onto1, onto2):
        """Расчет онтологической схожести из Этапа 1"""
        if not onto1 or not onto2:
            return 1.0

        # Простой расчет схожести на основе общих связей
        common = 0
        if 'source' in onto1 and 'source' in onto2:
            if onto1['source'] == onto2['source'] or onto1['source'] == onto2['target']:
                common += 1
        if 'target' in onto1 and 'target' in onto2:
            if onto1['target'] == onto2['target'] or onto1['target'] == onto2['source']:
                common += 1

        return 1 + common * 0.2


class LeninWorldviewTemplateBuilder:
    """Финальная версия построителя шаблона мировоззрения"""

    def __init__(self, lenin_pss_path, references_path, output_dir,
                 stage1_path, stage2_path):
        self.lenin_pss_path = lenin_pss_path
        self.references_path = references_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Инициализация с интеграцией первых двух этапов
        self.historical_context = HistoricalContext(stage1_path, stage2_path)
        self.preprocessor = TextPreprocessor(self.historical_context)
        self.stylometric_analyzer = StylometricAnalyzer()
        self.conceptual_builder = ConceptualNetworkBuilder(
            self.preprocessor,
            self.historical_context
        )

        # Данные
        self.lenin_texts = []
        self.lenin_dates = []
        self.reference_texts = []
        self.reference_authors = []
        self.reference_dates = []

        # Результаты
        self.full_conceptual_network = None
        self.worldview_template = {}

    def load_texts(self):
        """Загрузка текстов с улучшенной обработкой"""
        logger.info("Загрузка текстов Ленина")
        self.load_lenin_texts()

        logger.info("Загрузка референсных текстов")
        self.load_reference_texts()

    def load_lenin_texts(self):
        """Загрузка текстов Ленина с обработкой ошибок"""
        self.lenin_texts = []
        self.lenin_dates = []

        for vol_file in sorted(os.listdir(self.lenin_pss_path)):
            if vol_file.endswith('.txt'):
                vol_path = os.path.join(self.lenin_pss_path, vol_file)
                try:
                    with open(vol_path, 'r', encoding='utf-8', errors='replace') as f:
                        text = f.read()
                        if len(text.strip()) > 1000:  # Пропускаем слишком короткие файлы
                            self.lenin_texts.append(text)

                            # Извлечение года
                            year_match = re.search(r'\d{4}', vol_file)
                            year = int(year_match.group()) if year_match else 1900
                            self.lenin_dates.append(year)
                        else:
                            logger.warning(f"Файл {vol_file} слишком короткий, пропущен")
                except Exception as e:
                    logger.error(f"Ошибка загрузки файла {vol_file}: {str(e)}")

    def load_reference_texts(self):
        """Загрузка референсных текстов с обработкой ошибок"""
        self.reference_texts = []
        self.reference_authors = []
        self.reference_dates = []

        for author_dir in os.listdir(self.references_path):
            author_path = os.path.join(self.references_path, author_dir)
            if os.path.isdir(author_path):
                for work_file in os.listdir(author_path):
                    if work_file.endswith('.txt'):
                        work_path = os.path.join(author_path, work_file)
                        try:
                            with open(work_path, 'r', encoding='utf-8', errors='replace') as f:
                                text = f.read()
                                if len(text.strip()) > 500:  # Пропускаем слишком короткие файлы
                                    self.reference_texts.append(text)
                                    self.reference_authors.append(author_dir)

                                    # Оценочная датировка для референсов
                                    year_match = re.search(r'\d{4}', work_file)
                                    year = int(year_match.group()) if year_match else 1850
                                    self.reference_dates.append(year)
                                else:
                                    logger.warning(f"Файл {work_file} слишком короткий, пропущен")
                        except Exception as e:
                            logger.error(f"Ошибка загрузки файла {work_file}: {str(e)}")

    def build_conceptual_network(self):
        """Построение интегрированной концептуальной сети"""
        logger.info("Построение концептуальной сети с историческим контекстом")
        all_texts = self.lenin_texts + self.reference_texts
        all_dates = self.lenin_dates + self.reference_dates
        all_authors = ["lenin"] * len(self.lenin_texts) + self.reference_authors

        self.full_conceptual_network = self.conceptual_builder.build_network(
            all_texts,
            all_dates,
            all_authors
        )

        # Визуализация сети
        self.visualize_network()

    def visualize_network(self):
        """Визуализация концептуальной сети"""
        if not self.full_conceptual_network:
            return

        plt.figure(figsize=(20, 15))

        # Раскраска узлов
        node_colors = []
        node_sizes = []
        labels = {}

        for node in self.full_conceptual_network.nodes:
            node_type = self.full_conceptual_network.nodes[node].get('type', '')
            if node_type == 'text' and 'lenin' in node:
                node_colors.append('red')
                node_sizes.append(100)
                labels[node] = 'Ленин'
            elif node_type == 'text':
                node_colors.append('blue')
                node_sizes.append(80)
                labels[node] = self.full_conceptual_network.nodes[node].get('author', '')
            else:
                node_colors.append('green')
                node_sizes.append(30)
                # Показываем только ключевые концепты
                if self.full_conceptual_network.degree[node] > 2:
                    labels[node] = node

        # Построение графа
        pos = nx.spring_layout(self.full_conceptual_network, k=0.15, iterations=50)
        nx.draw(
            self.full_conceptual_network,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color="gray",
            alpha=0.6
        )

        # Ручная подпись ключевых узлов
        for node, label in labels.items():
            if node in pos:
                plt.text(
                    pos[node][0],
                    pos[node][1],
                    label,
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7)
                )

        plt.title("Концептуальная сеть мировоззрения Ленина", fontsize=16)
        plt.savefig(os.path.join(self.output_dir, "lenin_worldview_network.png"))
        plt.close()

    def identify_unique_patterns(self):
        """Выявление уникальных паттернов с интеграцией первых этапов"""
        logger.info("Идентификация уникальных паттернов Ленина")

        # Сбор данных по Ленину из сети
        lenin_nodes = [n for n in self.full_conceptual_network.nodes
                       if self.full_conceptual_network.nodes[n].get('type') == 'text'
                       and self.full_conceptual_network.nodes[n].get('author') == 'lenin']

        lenin_concepts = set()
        for node in lenin_nodes:
            neighbors = list(self.full_conceptual_network.neighbors(node))
            lenin_concepts.update(neighbors)

        # Анализ уникальных концептов
        unique_patterns = []
        for concept in lenin_concepts:
            if self.full_conceptual_network.nodes[concept].get('type') != 'concept':
                continue

            # Данные из первых этапов
            transformation = self.historical_context.get_transformation_index(concept)
            ontological = self.historical_context.get_ontological_foundation(concept)

            # Связи в сети
            connections = []
            for neighbor in self.full_conceptual_network.neighbors(concept):
                if self.full_conceptual_network.nodes[neighbor].get('type') == 'concept':
                    weight = self.full_conceptual_network[concept][neighbor].get('weight', 0)
                    years = self.full_conceptual_network[concept][neighbor].get('years', [])
                    connections.append({
                        'concept': neighbor,
                        'weight': weight,
                        'years': sorted(list(set(years)))
                    })

            unique_patterns.append({
                'concept': concept,
                'transformation': transformation,
                'ontological_links': ontological,
                'connections': sorted(connections, key=lambda x: x['weight'], reverse=True)[:5]
            })

        # Формирование шаблона
        self.worldview_template = {
            'metadata': {
                'created': datetime.datetime.now().isoformat(),
                'lenin_volumes': len(self.lenin_texts),
                'reference_authors': list(set(self.reference_authors)),
                'historical_events': self.historical_context.key_events
            },
            'unique_patterns': sorted(unique_patterns, key=lambda x: x['transformation'], reverse=True),
            'evolution': self.analyze_evolution()
        }

    def analyze_evolution(self):
        """Анализ эволюции с интеграцией исторического контекста"""
        evolution = {}

        # Группировка по периодам
        periods = {
            (1890, 1905): "Ранний период",
            (1905, 1917): "Революционный период",
            (1917, 1924): "Послереволюционный период"
        }

        for (start, end), period_name in periods.items():
            period_texts = [text for text, year in zip(self.lenin_texts, self.lenin_dates)
                            if start <= year <= end]
            period_dates = [year for year in self.lenin_dates if start <= year <= end]

            if not period_texts:
                continue

            # Анализ концептов периода
            concepts = []
            for text, year in zip(period_texts, period_dates):
                concepts.extend(self.conceptual_builder.extract_key_terms(text, year, n=10))

            concept_counts = Counter(concepts)
            top_concepts = [concept for concept, count in concept_counts.most_common(10)]

            # Исторический контекст периода
            context_events = [e for year, e in self.historical_context.key_events.items()
                              if start <= year <= end]

            # Формирование данных периода
            evolution[period_name] = {
                'years': f"{start}-{end}",
                'top_concepts': top_concepts,
                'historical_events': context_events,
                'text_samples': len(period_texts)
            }

        return evolution

    def export_template(self):
        """Экспорт финального шаблона"""
        logger.info("Экспорт шаблона")
        with open(os.path.join(self.output_dir, "lenin_worldview_template.json"), 'w', encoding='utf-8') as f:
            json.dump(self.worldview_template, f, ensure_ascii=False, indent=2)
        logger.info(f"Шаблон сохранен в {self.output_dir}")

    def execute_pipeline(self):
        """Оптимизированный пайплайн выполнения"""
        try:
            start_time = datetime.datetime.now()

            self.load_texts()
            self.build_conceptual_network()
            self.identify_unique_patterns()
            self.export_template()

            duration = datetime.datetime.now() - start_time
            logger.info(f"Пайплайн успешно завершен за {duration}")
            return True

        except Exception as e:
            logger.error(f"Ошибка выполнения пайплайна: {str(e)}")
            return False


def combine_stage_data(ontology_path, specificity_path, output_path):
    """Объединение данных этапов 1 и 2"""
    # Загрузка данных этапа 1
    with open(ontology_path, 'r', encoding='utf-8') as f:
        ontology = json.load(f)

    # Загрузка данных этапа 2
    with open(specificity_path, 'r', encoding='utf-8') as f:
        specificity = json.load(f)

    # Создаем объединенную структуру
    combined = {
        "transformations": {},
        "interpretations": {},
        "relations": ontology.get("relations", []),
        "embeddings": {}
    }

    # Сопоставляем концепты
    all_concepts = set(ontology.get("concepts", []))
    analyzed_concepts = set(specificity.get("interpretations", {}).keys())

    # Добавляем данные из этапа 2
    for concept in analyzed_concepts:
        if concept in specificity.get("transformations", {}):
            combined["transformations"][concept] = {
                "transformation_index": specificity["transformations"][concept].get("transformation_index", 1.0),
                "occurrences": specificity["transformations"][concept].get("occurrences", 0)
            }

        if concept in specificity.get("interpretations", {}):
            combined["interpretations"][concept] = {
                "original_embedding": specificity["interpretations"][concept].get("original_embedding", [])
            }

    # Добавляем недостающие концепты из этапа 1
    for concept in all_concepts - analyzed_concepts:
        if concept in ontology.get("embeddings", {}):
            combined["embeddings"][concept] = ontology["embeddings"][concept]

            # Для непроанализированных концептов используем нейтральные значения
            combined["transformations"][concept] = {
                "transformation_index": 1.0,
                "occurrences": 0
            }

    # Сохраняем объединенные данные
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Объединенные данные сохранены в {output_path}")
    print(f"Всего концептов: {len(all_concepts)}")
    print(f"Проанализированных концептов: {len(analyzed_concepts)}")
    print(f"Добавленных концептов: {len(all_concepts - analyzed_concepts)}")


# Пример использования
if __name__ == "__main__":
    # Конфигурация путей
    LENIN_PSS_PATH = "data/raw"
    REFERENCES_PATH = "data/books"
    OUTPUT_DIR = "data/stage3/lenin_worldview"
    STAGE1_PATH = "data/stage1/foundation_ontology.json"
    STAGE2_PATH = "data/stage2/lenin_dialectical_specificity_top1000.json"
    COMBINED_PATH = "data/stage3/combined_worldview_data.json"

    # Объединяем данные этапов 1 и 2
    combine_stage_data(
        ontology_path=STAGE1_PATH,
        specificity_path=STAGE2_PATH,
        output_path=COMBINED_PATH
    )

    # Создаем и запускаем построитель шаблона
    builder = LeninWorldviewTemplateBuilder(
        lenin_pss_path=LENIN_PSS_PATH,
        references_path=REFERENCES_PATH,
        output_dir=OUTPUT_DIR,
        stage1_path=STAGE1_PATH,
        stage2_path=STAGE2_PATH
    )

    if builder.execute_pipeline():
        logger.info("Шаблон мировоззрения успешно создан")
    else:
        logger.error("Ошибка создания шаблона")