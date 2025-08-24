import os
import json
import re
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("worldview_builder.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Предварительная обработка текстов"""

    def __init__(self):
        self.stop_words = self.load_stop_words()

    def load_stop_words(self):
        """Загрузка стоп-слов для русского языка"""
        return set([
            "и", "в", "не", "на", "с", "по", "к", "а", "из", "от", "то",
            "что", "как", "но", "он", "я", "мы", "вы", "его", "ее", "их",
            "это", "тот", "этот", "такой", "где", "когда", "даже", "лишь",
            "уже", "или", "если", "чтобы", "хотя", "за", "до", "после"
        ])

    def clean_text(self, text):
        """Очистка и нормализация текста"""
        # Удаление специальных символов и цифр
        text = re.sub(r"[^а-яА-ЯёЁ\s\-]", "", text)
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление лишних пробелов
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        """Токенизация текста с фильтрацией стоп-слов"""
        tokens = text.split()
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]

    def preprocess(self, text):
        """Полный цикл предварительной обработки"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return tokens


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
    """Построение семантических сетей концептов"""

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.vectorizer = TfidfVectorizer(max_features=2000)  # Уменьшим количество фич
        self.svd = TruncatedSVD(n_components=50)  # Уменьшим количество компонент

    def extract_key_terms(self, text, n=20):
        """Извлечение ключевых терминов с помощью TF-IDF"""
        try:
            # Для одного текста
            if isinstance(text, str):
                tokens = self.preprocessor.preprocess(text)
                preprocessed_text = " ".join(tokens)
                if not preprocessed_text.strip():
                    return []

                # Для коротких текстов используем CountVectorizer
                if len(tokens) < 100:
                    vectorizer = CountVectorizer(max_features=n)
                else:
                    vectorizer = TfidfVectorizer(max_features=n)

                vectorizer.fit([preprocessed_text])
                return vectorizer.get_feature_names_out()

            # Для корпуса текстов
            processed_corpus = [" ".join(self.preprocessor.preprocess(t)) for t in text]
            processed_corpus = [t for t in processed_corpus if t.strip()]

            if not processed_corpus:
                logger.warning("Корпус текстов пуст после предобработки")
                return [], None

            self.vectorizer.fit(processed_corpus)
            terms = self.vectorizer.get_feature_names_out()

            # Уменьшение размерности
            tfidf_matrix = self.vectorizer.transform(processed_corpus)

            # Проверка на возможность уменьшения размерности
            if tfidf_matrix.shape[1] < self.svd.n_components:
                logger.warning(
                    f"Уменьшение размерности невозможно: features={tfidf_matrix.shape[1]}, n_components={self.svd.n_components}")
                return terms, tfidf_matrix

            try:
                reduced_matrix = self.svd.fit_transform(tfidf_matrix)
                return terms, reduced_matrix
            except ValueError as e:
                logger.warning(f"Ошибка SVD: {str(e)}. Используем исходную матрицу")
                return terms, tfidf_matrix

        except Exception as e:
            logger.error(f"Ошибка извлечения терминов: {str(e)}")
            return [], None

    def build_network(self, texts, author_labels=None):
        """Построение семантической сети с оптимизированным подходом"""
        logger.info("Построение концептуальной сети")
        try:
            # Извлечение терминов
            terms, doc_vectors = self.extract_key_terms(texts)

            # Проверка на пустые данные
            if len(terms) == 0 or doc_vectors is None:
                logger.warning("Не удалось извлечь данные для построения сети")
                return nx.Graph()

            # Создание графа
            graph = nx.Graph()

            # Добавление узлов (авторов/текстов)
            for i, text in enumerate(texts):
                author = author_labels[i] if author_labels else f"text_{i}"
                graph.add_node(author, type="text")

                # Извлечение терминов для текущего текста
                text_terms = self.extract_key_terms(text, n=15)
                for term in text_terms:
                    if term not in graph:
                        graph.add_node(term, type="concept")
                    graph.add_edge(author, term, weight=1.0)

            # Построение связей между терминами на основе совместной встречаемости
            logger.info("Построение связей между терминами")
            term_cooccurrence = defaultdict(lambda: defaultdict(int))

            for text in texts:
                text_terms = self.extract_key_terms(text, n=30)
                for i, term1 in enumerate(text_terms):
                    for term2 in text_terms[i + 1:]:
                        term_cooccurrence[term1][term2] += 1
                        term_cooccurrence[term2][term1] += 1

            # Добавление связей между терминами
            for term1, co_terms in term_cooccurrence.items():
                for term2, count in co_terms.items():
                    if term1 in graph and term2 in graph:
                        # Нормализация веса
                        weight = count / (len(term_cooccurrence[term1]) + len(term_cooccurrence[term2]))
                        graph.add_edge(term1, term2, weight=weight)

            return graph

        except Exception as e:
            logger.error(f"Ошибка построения сети: {str(e)}")
            return nx.Graph()

    def extract_lenin_specific_patterns(self, full_network, lenin_texts):
        """Выявление уникальных для Ленина концептуальных паттернов"""
        try:
            # Извлечение терминов Ленина
            lenin_terms = set()
            for text in lenin_texts:
                terms = self.extract_key_terms(text, n=20)
                lenin_terms.update(terms)

            # Проверка на пустые термины
            if not lenin_terms:
                logger.warning("Не удалось извлечь термины Ленина")
                return []

            # Фильтрация сети
            nodes_to_include = [n for n in full_network.nodes if
                                n in lenin_terms or full_network.nodes[n].get('type') == 'text']
            if not nodes_to_include:
                logger.warning("Нет узлов для включения в подграф Ленина")
                return []

            lenin_subgraph = full_network.subgraph(nodes_to_include)

            # Выявление уникальных связей
            unique_connections = []
            for u, v, data in lenin_subgraph.edges(data=True):
                # Учитываем только связи между концептами
                if lenin_subgraph.nodes[u].get('type') == 'concept' and \
                        lenin_subgraph.nodes[v].get('type') == 'concept':
                    # Проверка, насколько связь характерна для Ленина
                    weight = data.get('weight', 0)
                    if weight > 0.1:  # Более низкий порог
                        unique_connections.append((u, v, weight))

            return unique_connections

        except Exception as e:
            logger.error(f"Ошибка выявления паттернов Ленина: {str(e)}")
            return []


class EvolutionAnalyzer:
    """Анализ эволюции взглядов во времени"""

    def __init__(self, stylometric_analyzer, conceptual_builder):
        self.stylometric_analyzer = stylometric_analyzer
        self.conceptual_builder = conceptual_builder

    def analyze(self, texts, dates):
        """Трекер изменений стиля и концептов во времени"""
        # Сортировка текстов по дате
        sorted_indices = np.argsort(dates)
        sorted_texts = [texts[i] for i in sorted_indices]
        sorted_dates = [dates[i] for i in sorted_indices]

        evolution = {
            'stylometric': [],
            'conceptual': [],
            'key_terms': []
        }

        prev_style = None
        prev_terms = None

        for i, text in enumerate(sorted_texts):
            # Стилометрический анализ
            current_style = self.stylometric_analyzer.analyze(text)

            # Анализ концептов
            current_terms = set(self.conceptual_builder.extract_key_terms(text, n=20))

            # Расчет изменений
            if prev_style is not None:
                style_change = self.stylometric_analyzer.compare(prev_style, current_style)
                term_change = 1 - len(prev_terms & current_terms) / len(prev_terms | current_terms)

                evolution['stylometric'].append({
                    'date': sorted_dates[i],
                    'change': style_change,
                    'features': current_style
                })

                evolution['conceptual'].append({
                    'date': sorted_dates[i],
                    'change': term_change,
                    'new_terms': list(current_terms - prev_terms),
                    'lost_terms': list(prev_terms - current_terms)
                })

            prev_style = current_style
            prev_terms = current_terms

            # Сохранение ключевых терминов для каждого периода
            evolution['key_terms'].append({
                'date': sorted_dates[i],
                'terms': list(current_terms)
            })

        return evolution


class LeninWorldviewTemplateBuilder:
    """Основной класс для построения шаблона мировоззрения"""

    def __init__(self, lenin_pss_path, references_path, output_dir):
        self.lenin_pss_path = lenin_pss_path
        self.references_path = references_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Инициализация компонентов
        self.preprocessor = TextPreprocessor()
        self.stylometric_analyzer = StylometricAnalyzer()
        self.conceptual_builder = ConceptualNetworkBuilder(self.preprocessor)
        self.evolution_analyzer = EvolutionAnalyzer(
            self.stylometric_analyzer,
            self.conceptual_builder
        )

        # Данные
        self.lenin_texts = []
        self.lenin_dates = []
        self.reference_texts = []
        self.reference_authors = []

        # Результаты
        self.lenin_stylometric = {}
        self.reference_stylometric = {}
        self.full_conceptual_network = None
        self.lenin_evolution = {}
        self.worldview_template = {}

    def load_texts(self):
        """Загрузка текстов Ленина и референсных авторов"""
        logger.info("Загрузка текстов Ленина")
        self.load_lenin_texts()

        logger.info("Загрузка референсных текстов")
        self.load_reference_texts()

    def load_lenin_texts(self):
        """Загрузка полного собрания сочинений Ленина"""
        for vol_file in sorted(os.listdir(self.lenin_pss_path)):
            if vol_file.endswith('.txt'):
                vol_path = os.path.join(self.lenin_pss_path, vol_file)
                with open(vol_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    self.lenin_texts.append(text)

                    # Извлечение года из названия файла (том_1902.txt -> 1902)
                    year_match = re.search(r'\d{4}', vol_file)
                    year = int(year_match.group()) if year_match else 1900
                    self.lenin_dates.append(year)

    def load_reference_texts(self):
        """Загрузка работ других философов"""
        for author_dir in os.listdir(self.references_path):
            author_path = os.path.join(self.references_path, author_dir)
            if os.path.isdir(author_path):
                for work_file in os.listdir(author_path):
                    if work_file.endswith('.txt'):
                        work_path = os.path.join(author_path, work_file)
                        with open(work_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            self.reference_texts.append(text)
                            self.reference_authors.append(author_dir)

    def analyze_stylometry(self):
        """Стилометрический анализ текстов"""
        logger.info("Анализ стилистики Ленина")
        self.lenin_stylometric = {}
        for i, text in enumerate(tqdm(self.lenin_texts)):
            vol_id = f"lenin_vol_{i + 1}"
            self.lenin_stylometric[vol_id] = self.stylometric_analyzer.analyze(text)

        logger.info("Анализ стилистики референсных авторов")
        self.reference_stylometric = {}
        for i, text in enumerate(tqdm(self.reference_texts)):
            author = self.reference_authors[i]
            if author not in self.reference_stylometric:
                self.reference_stylometric[author] = []
            self.reference_stylometric[author].append(self.stylometric_analyzer.analyze(text))

    def build_conceptual_network(self):
        """Построение концептуальной сети"""
        logger.info("Построение концептуальной сети")
        all_texts = self.lenin_texts + self.reference_texts
        all_authors = ["lenin"] * len(self.lenin_texts) + self.reference_authors

        self.full_conceptual_network = self.conceptual_builder.build_network(
            all_texts,
            all_authors
        )

        # Визуализация сети
        self.visualize_network()

    def visualize_network(self):
        """Визуализация концептуальной сети"""
        plt.figure(figsize=(20, 15))

        # Раскраска узлов
        node_colors = []
        for node in self.full_conceptual_network.nodes:
            if node == "lenin":
                node_colors.append("red")
            elif node in self.reference_authors:
                node_colors.append("blue")
            else:
                node_colors.append("green")

        # Размер узлов
        node_sizes = [100 if node in ["lenin"] + self.reference_authors else 30
                      for node in self.full_conceptual_network.nodes]

        pos = nx.spring_layout(self.full_conceptual_network, k=0.15, iterations=50)
        nx.draw(
            self.full_conceptual_network,
            pos,
            with_labels=False,
            node_size=node_sizes,
            node_color=node_colors,
            edge_color="gray",
            alpha=0.6
        )

        # Подписи для авторов
        for author in ["lenin"] + self.reference_authors:
            if author in pos:
                plt.text(
                    pos[author][0],
                    pos[author][1],
                    author,
                    fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8)
                )

        plt.title("Концептуальная сеть философских текстов", fontsize=16)
        plt.savefig(os.path.join(self.output_dir, "conceptual_network.png"))
        plt.close()

    def analyze_evolution(self):
        """Анализ эволюции взглядов Ленина"""
        logger.info("Анализ эволюции взглядов Ленина")
        self.lenin_evolution = self.evolution_analyzer.analyze(
            self.lenin_texts,
            self.lenin_dates
        )

        # Визуализация эволюции
        self.visualize_evolution()

    def visualize_evolution(self):
        """Визуализация эволюции взглядов"""
        dates = [e['date'] for e in self.lenin_evolution['stylometric']]
        style_changes = [e['change'] for e in self.lenin_evolution['stylometric']]
        concept_changes = [e['change'] for e in self.lenin_evolution['conceptual']]

        plt.figure(figsize=(12, 6))

        plt.plot(dates, style_changes, 'b-o', label='Стилистические изменения')
        plt.plot(dates, concept_changes, 'r--s', label='Концептуальные изменения')

        plt.xlabel('Год')
        plt.ylabel('Уровень изменений')
        plt.title('Эволюция стиля и концептов Ленина')
        plt.legend()
        plt.grid(True)

        # Отметки ключевых событий
        key_events = {1905: 'Революция 1905', 1917: 'Октябрьская революция'}
        for year, event in key_events.items():
            if year in dates:
                idx = dates.index(year)
                plt.annotate(
                    event,
                    (dates[idx], max(style_changes[idx], concept_changes[idx])),
                    xytext=(10, 30),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->')
                )

        plt.savefig(os.path.join(self.output_dir, "lenin_evolution.png"))
        plt.close()

    def identify_unique_patterns(self):
        """Выявление уникальных паттернов мировоззрения Ленина"""
        logger.info("Идентификация уникальных паттернов Ленина")

        # 1. Стилистические особенности
        lenin_style_profile = self.aggregate_style_profile()
        ref_style_profiles = self.aggregate_reference_profiles()

        # 2. Концептуальные паттерны
        lenin_concepts = self.conceptual_builder.extract_lenin_specific_patterns(
            self.full_conceptual_network,
            self.lenin_texts
        )

        # 3. Эволюционные тренды
        evolution_trends = self.identify_evolution_trends()

        # Формирование шаблона
        self.worldview_template = {
            'metadata': {
                'created': datetime.datetime.now().isoformat(),
                'lenin_volumes': len(self.lenin_texts),
                'reference_authors': list(set(self.reference_authors))
            },
            'stylometric_profile': lenin_style_profile,
            'conceptual_patterns': lenin_concepts,
            'evolutionary_trends': evolution_trends,
            'comparison_with_references': ref_style_profiles
        }

    def aggregate_style_profile(self):
        """Создание усредненного стилистического профиля Ленина"""
        avg_profile = defaultdict(float)
        count = len(self.lenin_stylometric)

        for vol_profile in self.lenin_stylometric.values():
            for key, value in vol_profile.items():
                if isinstance(value, dict):
                    if key not in avg_profile:
                        avg_profile[key] = defaultdict(float)
                    for subkey, subvalue in value.items():
                        avg_profile[key][subkey] += subvalue / count
                else:
                    avg_profile[key] += value / count

        return dict(avg_profile)

    def aggregate_reference_profiles(self):
        """Сравнение с референсными авторами"""
        comparisons = {}
        lenin_avg = self.aggregate_style_profile()

        for author, profiles in self.reference_stylometric.items():
            author_avg = defaultdict(float)
            count = len(profiles)

            for profile in profiles:
                for key, value in profile.items():
                    if isinstance(value, dict):
                        if key not in author_avg:
                            author_avg[key] = defaultdict(float)
                        for subkey, subvalue in value.items():
                            author_avg[key][subkey] += subvalue / count
                    else:
                        author_avg[key] += value / count

            similarity = self.stylometric_analyzer.compare(
                lenin_avg,
                dict(author_avg)
            )

            comparisons[author] = {
                'similarity': similarity,
                'profile': dict(author_avg)
            }

        return comparisons

    def identify_evolution_trends(self):
        """Выявление основных эволюционных трендов"""
        trends = {
            'conceptual_shifts': [],
            'stylistic_changes': []
        }

        # Анализ концептуальных изменений
        concept_changes = self.lenin_evolution['conceptual']
        for i in range(1, len(concept_changes)):
            prev = concept_changes[i - 1]
            curr = concept_changes[i]

            if curr['change'] > 0.3:  # Значительное изменение
                trends['conceptual_shifts'].append({
                    'period': f"{prev['date']}-{curr['date']}",
                    'new_terms': curr['new_terms'],
                    'lost_terms': curr['lost_terms']
                })

        # Анализ стилистических изменений
        style_changes = self.lenin_evolution['stylometric']
        for i in range(1, len(style_changes)):
            prev = style_changes[i - 1]['features']
            curr = style_changes[i]['features']

            # Выявление значимых изменений
            significant_changes = {}
            for key in ['sentence_length', 'lexical_diversity', 'word_length']:
                change = abs(prev[key] - curr[key])
                if change > (0.1 * prev[key]):  # Изменение > 10%
                    significant_changes[key] = {
                        'from': prev[key],
                        'to': curr[key],
                        'change': change
                    }

            if significant_changes:
                trends['stylistic_changes'].append({
                    'period': f"{style_changes[i - 1]['date']}-{style_changes[i]['date']}",
                    'changes': significant_changes
                })

        return trends

    def export_template(self):
        """Экспорт шаблона мировоззрения"""
        logger.info("Экспорт шаблона")

        # Преобразование для сериализации
        serializable_template = json.loads(
            json.dumps(self.worldview_template, default=self.serialize))

        with open(os.path.join(self.output_dir, "lenin_worldview_template.json"), 'w', encoding='utf-8') as f:
            json.dump(serializable_template, f, ensure_ascii=False, indent=2)

        logger.info(f"Шаблон сохранен в {self.output_dir}")

    def serialize(self, obj):
        """Сериализация специальных объектов"""
        if isinstance(obj, defaultdict):
            return dict(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def execute_pipeline(self):
        """Полный пайплайн построения шаблона"""
        try:
            start_time = datetime.datetime.now()

            self.load_texts()
            self.analyze_stylometry()
            self.build_conceptual_network()
            self.analyze_evolution()
            self.identify_unique_patterns()
            self.export_template()

            duration = datetime.datetime.now() - start_time
            logger.info(f"Пайплайн успешно завершен за {duration}")
            return True

        except Exception as e:
            logger.error(f"Ошибка выполнения пайплайна: {str(e)}")
            return False


# Пример использования
if __name__ == "__main__":
    # Пути к данным (настроить в соответствии с вашей структурой)
    LENIN_PSS_PATH = "../data/raw"
    REFERENCES_PATH = "../data/books"
    OUTPUT_DIR = "../data/stage3/lenin_worldview"

    builder = LeninWorldviewTemplateBuilder(
        lenin_pss_path=LENIN_PSS_PATH,
        references_path=REFERENCES_PATH,
        output_dir=OUTPUT_DIR
    )

    if builder.execute_pipeline():
        logger.info("Шаблон мировоззрения успешно создан")
    else:
        logger.error("Ошибка создания шаблона")