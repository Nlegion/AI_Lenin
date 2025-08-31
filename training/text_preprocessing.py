import os
import re
import json
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import spacy
import subprocess
import sys
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import chardet


class PhilosophicalTextProcessor:
    def __init__(self):
        # Проверка и установка модели spaCy при необходимости
        try:
            self.nlp = spacy.load("ru_core_news_lg")
        except OSError:
            print("Установка модели spaCy ru_core_news_lg...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "ru_core_news_lg"])
            self.nlp = spacy.load("ru_core_news_lg")

        # Загрузка моделей transformers
        try:
            self.term_model = SentenceTransformer('cointegrated/LaBSE-en-ru')
            self.classifier_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
                "cointegrated/rubert-tiny", num_labels=5
            )
        except Exception as e:
            print(f"Ошибка загрузки моделей transformers: {e}")
            print("Попробуйте установить вручную: pip install transformers sentence-transformers")
            raise

        # Философский тезаурус
        self.philosophy_terms = self.load_philosophy_terms()
        self.lenin_specific_terms = ["партийность", "эмпириомонизм", "богостроительство", "махист"]

        # Шаблоны для структурирования
        self.structure_patterns = {
            "thesis": re.compile(r'(тезис[ы]?|положение[я]?)\s*[№\d]', re.I),
            "concept": re.compile(r'понятие\s+["«](.+?)["»]', re.I),
            "critique": re.compile(r'(критик[ау]|опровержение)\s+["«](.+?)["»]', re.I),
            "reference": re.compile(r'\((см\.|ср\.)\s*.+?\)', re.I)
        }

    def load_philosophy_terms(self):
        """Расширенный философский тезаурус"""
        return {
            "диалектика": ["диалектический метод", "единство противоположностей",
                           "отрицание отрицания", "развитие по спирали"],
            "материализм": ["исторический материализм", "диалектический материализм",
                            "теория отражения", "объективная реальность"],
            "познание": ["гносеология", "теория познания",
                         "практика как критерий истины", "субъективный образ"],
            "революция": ["классовая борьба", "диктатура пролетариата",
                          "социалистическая революция", "авангард партии"],
            "онтология": ["бытие", "сущность", "материя", "субстанция", "атрибут"]
        }

    def detect_document_type(self, filename):
        """Определение типа документа с улучшенной логикой"""
        filename_lower = filename.lower()

        if "письм" in filename_lower:
            return "письмо"
        if "конспект" in filename_lower or "тетрад" in filename_lower:
            return "конспект"
        if "критик" in filename_lower or "опровержен" in filename_lower:
            return "критика"
        if "капитал" in filename_lower or "экономи" in filename_lower:
            return "экономический трактат"
        if "философ" in filename_lower or "гносеолог" in filename_lower:
            return "философский труд"
        return "теоретический труд"

    def restore_notebook_structure(self, text):
        """Улучшенное восстановление структуры конспектов"""
        restored_text = []
        current_source = "Ленин"
        current_topic = ""

        # Разбивка на строки с сохранением пустых строк как разделителей
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if restored_text and restored_text[-1] != "\n":
                    restored_text.append("\n")
                continue

            # Определение источника
            source_match = re.search(r'(\b(Гегель|Маркс|Фейербах|Аристотель|Энгельс|Плеханов)\b)', line)
            if source_match:
                current_source = source_match.group(1)
                restored_text.append(f"\n\n[ИСТОЧНИК: {current_source}]\n\n")
                continue

            # Определение тем (заголовки в верхнем регистре)
            if re.match(r'^[А-ЯЁ\s\d]{10,80}$', line):
                current_topic = line
                restored_text.append(f"\n\n[ТЕМА: {current_topic}]\n\n")
                continue

            # Восстановление комментариев Ленина
            if re.match(r'^(NB!|Заметка|Примечание|Коммент|В.И.Ленин:)', line):
                restored_text.append(f"\n[КОММЕНТАРИЙ ЛЕНИНА]: {line}\n")
                continue

            # Объединение коротких строк в абзацы
            if len(line) < 120 and i + 1 < len(lines) and lines[i + 1].strip():
                restored_text.append(line + " ")
            else:
                restored_text.append(line + "\n")

        return "".join(restored_text)

    def restore_letter_structure(self, text):
        """Восстановление структуры писем"""
        # Выделение дат, адресатов и подписей
        text = re.sub(r'(\d{1,2}\s*[а-я]+\s*\d{4}\s*г\.)', r'\n[ДАТА: \1]\n', text)
        text = re.sub(r'(Уважаемый\s+[А-Я][а-я]+!|Дорогой\s+[А-Я][а-я]+!)', r'\n[ОБРАЩЕНИЕ: \1]\n', text)
        text = re.sub(r'(С\s+товарищеским\s+приветом[,\s]*|С\s+коммунистическим\s+приветом[,\s]*)',
                      r'\n[ПОДПИСЬ: \1]\n', text)
        return text

    def restore_critique_structure(self, text):
        """Восстановление структуры критических работ"""
        # Выделение цитат оппонентов
        text = re.sub(r'([«"])(.+?)([»"])\s*—\s*писал\s+([А-Я][а-я]+)',
                      r'\n[ЦИТАТА \4]: «\2»\n', text)

        # Выделение тезисов для критики
        text = re.sub(r'(Он\s+утверждает,? что|По\s+его\s+мнению,?)',
                      r'\n[ТЕЗИС ОППОНЕНТА]: \1', text)
        return text

    def identify_conceptual_chains(self, text):
        """Улучшенное выявление концептуальных цепочек"""
        # Используем spaCy только для больших текстов
        if len(text) > 1000000:  # Ограничение для больших текстов
            return {}

        try:
            doc = self.nlp(text)
            conceptual_chains = defaultdict(list)
            current_concept = None

            for sent in doc.sents:
                # Проверяем наличие основных терминов в предложении
                found = False
                for concept, terms in self.philosophy_terms.items():
                    concept_terms = [concept] + terms
                    if any(term in sent.text.lower() for term in concept_terms):
                        current_concept = concept
                        conceptual_chains[concept].append(sent.text)
                        found = True
                        break

                # Если не нашли новый концепт, продолжаем текущий
                if not found and current_concept:
                    conceptual_chains[current_concept].append(sent.text)

            return dict(conceptual_chains)
        except Exception as e:
            print(f"Ошибка обработки текста: {e}")
            return {}

    def semantic_cleanup(self, text):
        """Улучшенная семантическая очистка текста"""
        # Автоматическое исправление частых OCR-ошибок
        corrections = {
            r'\b[Дд][и][аа][л][ее][кк][тт][ии][кк][ии]\b': 'диалектики',
            r'\b[Мм][аа][тт][ее][рр][ии][аа][лл][ии][зз][м]\b': 'материализм',
            r'\b[Эе][м][п][ии][р][ии][оо][ -]?[кк][р][ии][тт][ии][цц][ии][зз][м]\b': 'эмпириокритицизм',
            r'\b[Пп][аа][рр][тт][ии][ий][нн][оо][с][тт][ь]\b': 'партийность',
            r'\b[Гг][е][г][е][л][ья]\b': 'Гегеля',
            r'\b[Лл][ее][нн][ии][нн][аа]\b': 'Ленина',
            r'\b[Мм][аа][рр][кк][с][с][аа]\b': 'Маркса',
        }

        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)

        # Удаление номеров страниц и колонтитулов
        text = re.sub(r'\bСтр?\.?\s*\d+\b', '', text)
        text = re.sub(r'\bТом\s+[IVXL]+\b', '', text, flags=re.I)

        return text

    def process_file(self, file_path):
        """Обработка файла с улучшенной обработкой ошибок"""
        try:
            # Автоматическое определение кодировки
            with open(file_path, 'rb') as f:
                raw_data = f.read(50000)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'

                # Проверка проблемных кодировок
                if encoding.lower() in ['windows-1251', 'cp1251']:
                    encoding = 'cp1251'
                elif encoding.lower() == 'iso-8859-5':
                    encoding = 'iso-8859-5'
                else:
                    encoding = 'utf-8'

            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                text = f.read()

        except Exception as e:
            print(f"Ошибка чтения файла {file_path}: {e}")
            return None

        filename = os.path.basename(file_path)

        # Применение семантической очистки
        text = self.semantic_cleanup(text)

        # Восстановление структуры в зависимости от типа документа
        doc_type = self.detect_document_type(filename)
        if doc_type == "конспект":
            text = self.restore_notebook_structure(text)
        elif doc_type == "письмо":
            text = self.restore_letter_structure(text)
        elif doc_type == "критика":
            text = self.restore_critique_structure(text)

        # Извлечение концептуальных цепочек
        conceptual_chains = self.identify_conceptual_chains(text)

        # Формат результата
        return {
            "metadata": {
                "filename": filename,
                "document_type": doc_type,
                "concepts": list(conceptual_chains.keys())
            },
            "content": text,
            "conceptual_chains": conceptual_chains,
        }


def process_corpus(input_dir, output_dir):
    """Обработка корпуса с обработкой ошибок"""
    processor = PhilosophicalTextProcessor()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Сбор всех текстовых файлов
    txt_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, file))

    print(f"Найдено текстовых файлов: {len(txt_files)}")

    if not txt_files:
        print("Ошибка: не найдено ни одного файла для обработки")
        print("Проверьте путь к директории с книгами:", input_dir)
        print("Структура директории должна соответствовать указанной в задании")
        return

    processed_count = 0
    for file_path in tqdm(txt_files, desc="Интеллектуальная обработка"):
        result = processor.process_file(file_path)
        if not result:
            continue

        # Создание относительного пути для сохранения
        rel_path = os.path.relpath(file_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path + '.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            processed_count += 1
        except Exception as e:
            print(f"Ошибка сохранения {output_path}: {e}")

    print(f"Успешно обработано файлов: {processed_count}/{len(txt_files)}")


if __name__ == "__main__":
    # Проверка и установка необходимых пакетов
    required_packages = {
        "spacy": "spacy",
        "transformers": "transformers",
        "sentence-transformers": "sentence-transformers",
        "chardet": "chardet",
        "tqdm": "tqdm",
        "torch": "torch"
    }

    missing_packages = []
    for package, install_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(install_name)

    if missing_packages:
        print(f"Установка недостающих пакетов: {', '.join(missing_packages)}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)

    # Проверка модели spaCy
    try:
        import spacy

        spacy.load("ru_core_news_lg")
    except:
        print("Установка модели spaCy ru_core_news_lg...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "ru_core_news_lg"])

    # Пути обработки
    input_directory = "books"
    output_directory = "processed_intellectual"

    # Проверка существования входной директории
    if not os.path.exists(input_directory):
        print(f"Ошибка: директория '{input_directory}' не существует")
        print("Создайте директорию 'books' и поместите в неё текстовые файлы")
        print("Структура должна соответствовать указанной в задании:")
        print("books/")
        print("  Аристотель/")
        print("    Метафизика (Классики философии) - 1934.txt")
        print("  Гегель/")
        print("    Наука логики.txt")
        print("  ... и т.д.")
    else:
        process_corpus(input_directory, output_directory)
        print(f"Обработка завершена. Результаты в: {output_directory}")